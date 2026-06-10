"""IVR (Iterative Visual Retracing) 推理主逻辑"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from PIL import Image

from .confidence import ConfidenceEstimator
from .roi_extractor import ROIExtractor


class IVRInference:
    """迭代视觉追溯推理引擎

    输入图像 + 问题 → 多次回看不同区域 → 迭代验证 → 最终答案
    """

    def __init__(
        self,
        model,
        processor,
        config,
    ):
        """
        Args:
            model: MiniCPM-V-4.6 模型（冻结）
            processor: 对应的 processor
            config: Config 对象
        """
        self.model = model
        self.processor = processor
        self.config = config

        self.max_passes = config.max_passes
        self.conf_high = config.confidence_high
        self.conf_low = config.confidence_low

        self.confidence_estimator = ConfidenceEstimator()
        self.roi_extractor = ROIExtractor(
            strategy=config.ivr.get("roi_strategy", "attention"),
            num_rois=config.ivr.get("num_rois", 3),
        )

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        """
        Args:
            image: PIL Image
            question: 自然语言问题

        Returns:
            {
                "answer": str,
                "num_passes": int,
                "pass_answers": [str, ...],    # 每轮输出
                "pass_confidences": [float, ...],  # 每轮置信度
                "rois": [bbox, ...],           # 聚焦的 ROI
                "final_confidence": float,
                "final_action": str,           # "stop" | "max_passes_reached"
            }
        """
        trace = {
            "pass_answers": [],
            "pass_confidences": [],
            "rois": [],
        }

        # ── Pass 1: 全局浏览 ──
        answer_1, logits_1 = self._single_pass(image, question)
        conf_1 = self.confidence_estimator.compute(
            output_ids=answer_1["output_ids"],
            logits=logits_1,
        )
        trace["pass_answers"].append(answer_1["text"])
        trace["pass_confidences"].append(conf_1)

        action = self.confidence_estimator.should_retry(
            conf_1, self.conf_high, self.conf_low
        )
        if action == "stop":
            return self._build_result(answer_1["text"], 1, "stop", trace)

        # ── Pass 2+: 迭代追溯 ──
        current_answer = answer_1["text"]
        current_region = None  # 默认全图

        for pass_i in range(2, self.max_passes + 1):
            # 提取 ROI
            if pass_i == 2:
                # 第一次追溯：基于 attention 找 ROI
                rois = self.roi_extractor.extract(
                    visual_features=answer_1.get("visual_features"),
                    answer_text=current_answer,
                    attentions=answer_1.get("attentions"),
                )
            else:
                # 后续：根据最新置信度决定是否换区域
                rois = self._update_rois(prev_rois, action)

            if rois:
                current_region = rois[0]
                trace["rois"].append(current_region)

            # 聚焦区域追问
            refine_question = self._build_refine_question(
                question, current_answer, current_region, pass_i
            )
            if current_region:
                # 裁剪 ROI 区域作为额外视觉输入
                roi_image = image.crop(current_region)
                answer_i, logits_i = self._single_pass(
                    image, refine_question, roi_image=roi_image
                )
            else:
                answer_i, logits_i = self._single_pass(image, refine_question)

            conf_i = self.confidence_estimator.compute(
                output_ids=answer_i["output_ids"],
                logits=logits_i,
            )
            trace["pass_answers"].append(answer_i["text"])
            trace["pass_confidences"].append(conf_i)

            action = self.confidence_estimator.should_retry(
                conf_i, self.conf_high, self.conf_low
            )
            current_answer = answer_i["text"]

            if action == "stop":
                return self._build_result(current_answer, pass_i, "stop", trace)

        # ── 达到最大 pass 数 ──
        final_answer = current_answer + "\n\n[建议人工复核]"
        return self._build_result(
            final_answer, self.max_passes, "max_passes_reached", trace
        )

    def _single_pass(
        self,
        image: Image.Image,
        question: str,
        roi_image: Optional[Image.Image] = None,
    ) -> Tuple[Dict, Optional[torch.Tensor]]:
        """单次 VLM 推理"""
        # 构造 message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        if roi_image is not None:
            messages[0]["content"].insert(
                1, {"type": "text", "text": "\n[聚焦区域如下]\n"}
            )
            messages[0]["content"].insert(
                2, {"type": "image", "image": roi_image}
            )

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
        )

        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]

        answer_text = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        result = {
            "text": answer_text,
            "output_ids": output_ids,
        }
        return result, None

    def _build_refine_question(
        self,
        original_question: str,
        current_answer: str,
        region: Optional[Tuple] = None,
        pass_num: int = 2,
    ) -> str:
        """构造细化追问"""
        parts = [
            f"原始问题: {original_question}",
            f"你之前的判断: {current_answer}",
            f"请进行第{pass_num}次重新检查。",
        ]
        if region:
            parts.append(
                f"请特别关注图像中坐标({region[0]},{region[1]})到({region[2]},{region[3]})的区域。"
            )
        parts.append("如果发现之前遗漏或错误的地方，请修正。否则确认之前的判断。")
        return "\n".join(parts)

    def _update_rois(self, prev_rois: List, action: str) -> List:
        """更新 ROI 列表"""
        if action == "switch":
            # 轮转到下一个 ROI
            return prev_rois[1:] + prev_rois[:1]
        return prev_rois  # refine: 保持当前 ROI

    def _build_result(self, answer, passes, action, trace) -> Dict:
        return {
            "answer": answer,
            "num_passes": passes,
            "pass_answers": trace["pass_answers"],
            "pass_confidences": trace["pass_confidences"],
            "rois": trace.get("rois", []),
            "final_confidence": (
                trace["pass_confidences"][-1]
                if trace["pass_confidences"]
                else 0.0
            ),
            "final_action": action,
        }
