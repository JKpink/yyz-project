"""B2: CoT Prompting Baseline —— 提示模型逐步思考"""

import torch
from typing import Dict
from PIL import Image


class BaselineCoT:
    """CoT 提示：标准 chain-of-thought"""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    COT_PREFIX = "请一步步思考后再回答。"

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        cot_question = f"{self.COT_PREFIX}\n\n问题：{question}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": cot_question},
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
            max_new_tokens=384,  # CoT 需要更多 token
            temperature=0.2,
            do_sample=False,
        )

        answer = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        return {
            "answer": answer,
            "num_passes": 1,  # CoT 不增加视觉 pass
        }
