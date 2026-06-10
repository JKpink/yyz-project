"""B1: 直接回答 Baseline —— 标准 VLM 推理"""

import torch
from typing import Dict
from PIL import Image


class BaselineDirect:
    """标准 VLM：图像 + 问题 → 一步回答"""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
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

        return {
            "answer": answer,
            "num_passes": 1,
        }
