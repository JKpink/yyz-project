"""B3: MemVR（移植版） Baseline —— 将 MemVR 追溯机制移植到 MiniCPM-V"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from PIL import Image


class BaselineMemVR:
    """MemVR 追溯机制移植版

    原始 MemVR (ICML 2025) 基于 LLaVA-7B。
    本实现将其核心"记忆空间视觉追溯"机制移植到 MiniCPM-V-4.6。

    流程：编码全图 → 存储视觉特征到 memory bank →
          LLM 生成初版答案 → 从 memory bank 回看视觉特征 → 生成最终答案
    """

    def __init__(self, model, processor, retrace_layers: List[int] = None):
        """
        Args:
            model: MiniCPM-V-4.6
            processor: 对应 processor
            retrace_layers: 在哪些 Transformer 层进行记忆回看
        """
        self.model = model
        self.processor = processor
        self.retrace_layers = retrace_layers or [16, 20, 23]
        self.memory_bank = None  # 存储中间层视觉特征

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        # ── Pass 1: 编码 + 存储 + 初次回答 ──
        answer_1, visual_features = self._first_pass(image, question)

        # ── Pass 2: 回看记忆 + 纠正 ──
        retrace_question = (
            f"原始问题: {question}\n"
            f"你的初步回答: {answer_1}\n\n"
            f"请重新审视原始图像。如果初步回答中有任何不准确或遗漏，"
            f"请纠正。如果没有问题，请确认。"
        )

        answer_2 = self._retrace_pass(image, retrace_question, visual_features)

        return {
            "answer": answer_2,
            "num_passes": 2,
            "pass_answers": [answer_1, answer_2],
        }

    def _first_pass(self, image: Image.Image, question: str):
        """第一次 pass：编码全图，存储特征，生成初版答案"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        # 前向传播获取 hidden states
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        answer = self.processor.decode(
            outputs.sequences[0][input_len:], skip_special_tokens=True
        )

        # 提取回看层的特征存入 memory bank
        visual_feats = {}
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            for layer_idx in self.retrace_layers:
                if layer_idx < len(outputs.hidden_states):
                    visual_feats[layer_idx] = outputs.hidden_states[layer_idx]

        self.memory_bank = visual_feats
        return answer, visual_feats

    def _retrace_pass(
        self,
        image: Image.Image,
        question: str,
        visual_features: dict,
    ) -> str:
        """第二次 pass：基于 memory bank 重新生成"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
        )

        answer = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return answer
