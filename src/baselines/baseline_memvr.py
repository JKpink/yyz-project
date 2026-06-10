"""B3: MemVR（两阶段推理——先回答，再回看图纠正）

不做 monkey-patch，保持模型原封不动。
第一轮正常回答 → 第二轮带纠正 prompt 重看原图。
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
        # 第一轮：正常回答
        msg1 = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}]}]
        prompt1 = self.processor.apply_chat_template(
            msg1, tokenize=False, add_generation_prompt=True)
        inp1 = self.processor(text=prompt1, images=[image], return_tensors="pt")
        inp1 = {k: v.to(self.model.device) for k, v in inp1.items()}
        il1 = inp1["input_ids"].shape[-1]
        out1 = self.model.generate(**inp1, max_new_tokens=256, temperature=0.2, do_sample=False)
        if isinstance(out1, tuple): out1 = out1[0]
        first = self.processor.decode(out1[0][il1:], skip_special_tokens=True)

        # 第二轮：带着第一轮答案再审视原图
        msg2 = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"原始问题: {question}\n"
                f"你的初步回答: {first}\n\n"
                f"请重新仔细观察原始图像。如果初步回答中有任何错误或遗漏，"
                f"请纠正并给出最终答案。如果没有问题，请确认初步回答。"
            )}]}]
        prompt2 = self.processor.apply_chat_template(
            msg2, tokenize=False, add_generation_prompt=True)
        inp2 = self.processor(text=prompt2, images=[image], return_tensors="pt")
        inp2 = {k: v.to(self.model.device) for k, v in inp2.items()}
        il2 = inp2["input_ids"].shape[-1]
        out2 = self.model.generate(**inp2, max_new_tokens=256, temperature=0.2, do_sample=False)
        if isinstance(out2, tuple): out2 = out2[0]
        final = self.processor.decode(out2[0][il2:], skip_special_tokens=True)

        return {"answer": final, "num_passes": 2}
