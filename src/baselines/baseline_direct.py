"""B1: 直接回答 Baseline —— 官方 MiniCPM-V-4.6 API"""

import torch
from typing import Dict
from PIL import Image


class BaselineDirect:

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            downsample_mode="16x",
        ).to(self.model.device)
        il = inputs.input_ids.shape[-1]
        out = self.model.generate(**inputs, downsample_mode="16x", max_new_tokens=256)
        if isinstance(out, tuple): out = out[0]
        ans = self.processor.decode(out[0][il:], skip_special_tokens=True)
        return {"answer": ans, "num_passes": 1}
