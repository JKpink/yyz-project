"""IVR (Iterative Visual Retracing) 推理主逻辑"""

import torch, traceback
from typing import Dict, List, Tuple, Optional
from PIL import Image
from .confidence import ConfidenceEstimator
from .roi_extractor import ROIExtractor


class IVRInference:

    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.config = config
        self.max_passes = config.max_passes
        self.conf_high = config.confidence_high
        self.conf_low = config.confidence_low
        self.confidence_estimator = ConfidenceEstimator()
        self.roi_extractor = ROIExtractor(
            strategy=config.ivr.get("roi_strategy", "grid"),
            num_rois=config.ivr.get("num_rois", 3))

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        try:
            return self._impl(image, question)
        except Exception:
            traceback.print_exc()
            return {"answer": f"ERROR: {traceback.format_exc()}", "num_passes": 0}

    def _impl(self, image: Image.Image, question: str) -> Dict:
        trace = {"pass_answers": [], "pass_confidences": [], "rois": []}

        # Pass 1: 全局浏览
        a1, _ = self._single_pass(image, question)
        c1 = self.confidence_estimator.compute(output_ids=a1["output_ids"])
        trace["pass_answers"].append(a1["text"])
        trace["pass_confidences"].append(c1)
        act = self.confidence_estimator.should_retry(c1, self.conf_high, self.conf_low)
        if act == "stop":
            return self._result(a1["text"], 1, "stop", trace)

        # Pass 2+
        cur = a1["text"]
        reg = None
        for pi in range(2, self.max_passes + 1):
            rois = (self.roi_extractor.extract(None, cur) if pi == 2
                    else self._rotate_rois(prev_rois, act))
            prev_rois = rois
            if rois:
                reg = rois[0]
                trace["rois"].append(reg)
            q = self._refine(question, cur, reg, pi)
            ai, _ = self._single_pass(image, q)
            ci = self.confidence_estimator.compute(output_ids=ai["output_ids"])
            trace["pass_answers"].append(ai["text"])
            trace["pass_confidences"].append(ci)
            act = self.confidence_estimator.should_retry(ci, self.conf_high, self.conf_low)
            cur = ai["text"]
            if act == "stop":
                return self._result(cur, pi, "stop", trace)

        return self._result(cur + "\n\n[建议人工复核]", self.max_passes, "max", trace)

    def _single_pass(self, image, question):
        msg = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}]}]
        inp = self.processor.apply_chat_template(
            msg, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", downsample_mode="16x"
        ).to(self.model.device)
        il = inp.input_ids.shape[-1]
        out = self.model.generate(**inp, downsample_mode="16x", max_new_tokens=256)
        if isinstance(out, tuple): out = out[0]
        txt = self.processor.decode(out[0][il:], skip_special_tokens=True)
        return {"text": txt, "output_ids": out}, None

    def _refine(self, q, ans, reg, n):
        p = [f"原始问题: {q}", f"之前的判断: {ans}",
             f"第{n}次复查。请仔细观察原图。"]
        if reg:
            p.append(f"请特别关注坐标({reg[0]},{reg[1]})-({reg[2]},{reg[3]})区域。")
        p.append("纠正错误或遗漏，或确认之前判断。")
        return "\n".join(p)

    def _rotate_rois(self, prev, act):
        if act == "switch": return prev[1:] + prev[:1]
        return prev

    def _result(self, a, p, act, t):
        return {"answer": a, "num_passes": p,
                "pass_answers": t["pass_answers"],
                "pass_confidences": t["pass_confidences"],
                "rois": t["rois"],
                "final_confidence": t["pass_confidences"][-1] if t["pass_confidences"] else 0,
                "final_action": act}
