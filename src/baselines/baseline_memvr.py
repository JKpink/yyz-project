"""B3: MemVR（PyTorch hook 版）

用 register_forward_hook 在每层监控熵值，注入视觉特征。
hooks 自动 add/remove，无残留问题。
"""

import torch, traceback
import torch.nn.functional as F
import numpy as np
from typing import Dict
from PIL import Image


class BaselineMemVR:

    def __init__(self, model, processor,
                 starting_layer=5, ending_layer=16,
                 entropy_threshold=0.75, retracing_ratio=0.12):
        self.model = model
        self.processor = processor
        self.starting_layer = starting_layer
        self.ending_layer = ending_layer
        self.entropy_threshold = entropy_threshold
        self.retracing_ratio = retracing_ratio
        self._hooks = []
        self._injected = False
        self._visual_token = None

    def _make_hook(self, layer_idx):
        patch = self
        def hook(module, input, output):
            if patch._injected:
                return output
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            # 计算 top-10 熵
            last_tok = hs[0, -1, :]
            logits = patch.model.lm_head(last_tok.to(
                next(patch.model.lm_head.parameters()).dtype))
            top_vals = torch.topk(logits.float(), 10).values
            probs = F.softmax(top_vals, dim=-1)
            ent = (-probs * torch.log(probs + 1e-10)).sum().item() / np.log(10)

            if (patch.starting_layer <= layer_idx <= patch.ending_layer
                    and ent > patch.entropy_threshold
                    and patch._visual_token is not None):
                patch._injected = True
                vt = patch._visual_token.to(device=hs.device, dtype=hs.dtype)
                # 取视觉特征均值注入每个 token
                inj = vt.mean(dim=1, keepdim=True).expand(-1, hs.shape[1], -1)
                hs_new = hs * (1 - patch.retracing_ratio) + inj * patch.retracing_ratio
                if isinstance(output, tuple):
                    return (hs_new,) + output[1:]
                return hs_new
            return output
        return hook

    def _patch(self, visual_token):
        self._unpatch()
        self._visual_token = visual_token
        self._injected = False
        for i, layer in enumerate(self.model.model.language_model.layers):
            h = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

    def _unpatch(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._visual_token = None
        self._injected = False

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        try:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}]}]
            inp = self.processor.apply_chat_template(
                msg, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", downsample_mode="16x"
            ).to(self.model.device)

            # 提取视觉 token
            pv = inp.get("pixel_values")
            vt = (self.model.model.vision_tower(pv).last_hidden_state
                  if pv is not None and self.model.model.vision_tower
                  else torch.zeros(1, 576, 1152, device=self.model.device, dtype=torch.float16))

            self._patch(vt)
            il = inp.input_ids.shape[-1]
            out = self.model.generate(**inp, downsample_mode="16x", max_new_tokens=256)
            if isinstance(out, tuple): out = out[0]
            ans = self.processor.decode(out[0][il:], skip_special_tokens=True)
            self._unpatch()
            return {"answer": ans, "num_passes": 2}
        except Exception:
            traceback.print_exc()
            self._unpatch()
            return {"answer": f"ERROR: {traceback.format_exc()}", "num_passes": 0}
