"""B2: Thinking 模型 Baseline —— MiniCPM-V-4.6-Thinking 内置推理链"""

import torch
from typing import Dict
from PIL import Image


class BaselineThinking:
    """使用内置 Thinking 能力的模型

    基座: openbmb/MiniCPM-V-4.6-Thinking
    区别于 B1: 模型训练时学会了推理模式, 自动生成思考过程再回答
    """

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
            max_new_tokens=384,
            temperature=0.2,
            do_sample=False,
        )

        answer = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        return {
            "answer": answer,
            "num_passes": 1,  # Thinking 是文本推理, 不增加视觉 pass
        }
