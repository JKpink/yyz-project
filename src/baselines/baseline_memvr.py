"""B3: MemVR（两轮推理版）

不做 monkey-patch。第一轮正常回答 → 第二轮带纠正 prompt 再审视原图。
与 B4 (IVR) 的区别: B3 固定回看 2 次全图, B4 自适应 2-4 次 + ROI 聚焦。
"""

import torch
from typing import Dict
from PIL import Image


class BaselineMemVR:

    def __init__(self, model, processor, **kwargs):
        self.model = model
        self.processor = processor

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        # 第一轮
        msg1 = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}]}]
        inp1 = self.processor.apply_chat_template(
            msg1, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", downsample_mode="16x"
        ).to(self.model.device)
        il1 = inp1.input_ids.shape[-1]
        o1 = self.model.generate(**inp1, downsample_mode="16x", max_new_tokens=256)
        if isinstance(o1, tuple): o1 = o1[0]
        first = self.processor.decode(o1[0][il1:], skip_special_tokens=True)

        # 第二轮（带纠正 prompt）
        msg2 = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"原始问题: {question}\n你的初步回答: {first}\n\n"
                f"请重新仔细观察原始图像。纠正错误或遗漏，或确认初步回答。"
            )}]}]
        inp2 = self.processor.apply_chat_template(
            msg2, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", downsample_mode="16x"
        ).to(self.model.device)
        il2 = inp2.input_ids.shape[-1]
        o2 = self.model.generate(**inp2, downsample_mode="16x", max_new_tokens=256)
        if isinstance(o2, tuple): o2 = o2[0]
        final = self.processor.decode(o2[0][il2:], skip_special_tokens=True)

        return {"answer": final, "num_passes": 2}
