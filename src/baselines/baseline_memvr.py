"""B3: MemVR（在 forward 层注入视觉特征，不修改 MLP）

架构: model.model.language_model.layers[i] (Qwen3_5DecoderLayer, 24层)
"""

import torch
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
        self._patched = False

    def _patch(self, visual_token):
        if self._patched:
            return
        patch = self
        N = len(self.model.model.language_model.layers)

        # 只替换 model.forward，在 hidden states 层面注入视觉特征
        patch._orig_fwd = self.model.forward

        def new_fwd(self_m, input_ids=None, attention_mask=None, position_ids=None,
                     past_key_values=None, inputs_embeds=None, pixel_values=None,
                     image_sizes=None, use_cache=None, output_attentions=None,
                     output_hidden_states=None, return_dict=None, cache_position=None,
                     **kw):
            out = patch._orig_fwd(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                image_sizes=image_sizes, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True, return_dict=True,
                cache_position=cache_position, **kw)
            hs = out.hidden_states
            if hs is None:
                return out

            done = False
            for layer_i in range(patch.starting_layer, min(patch.ending_layer, len(hs)-1)):
                # 计算当前层熵值
                last_tok = hs[layer_i][:, -1, :].float()
                logits = self_m.lm_head(last_tok)
                top_vals = torch.topk(logits, 10).values.float()
                probs = F.softmax(top_vals, dim=-1)
                ent = (-probs * torch.log(probs + 1e-10)).sum().item() / np.log(10)

                if not done and ent > patch.entropy_threshold:
                    done = True
                    # 直接把视觉特征注入到 hidden state（加权混合）
                    vt = visual_token.to(device=hs[layer_i].device, dtype=hs[layer_i].dtype)
                    # 对齐维度: visual_token [B,Nv,D] → 取平均得到 [B,1,D] → 加到每个 token
                    vt_avg = vt.mean(dim=1, keepdim=True)  # [B,1,D]
                    # 注入到所有 token 的 hidden state
                    hs_next = hs[layer_i + 1]
                    inj = vt_avg.expand(-1, hs_next.shape[1], -1)
                    # 替换最后一层的 hidden states（模拟"再看一眼"）
                    hs[layer_i + 1] = hs_next * (1 - patch.retracing_ratio) + inj * patch.retracing_ratio

            # 更新 output 的 last_hidden_state
            out.last_hidden_state = hs[-1]
            return out

        self.model.forward = new_fwd.__get__(self.model, type(self.model))
        self._patched = True

    def _unpatch(self):
        if not self._patched:
            return
        if hasattr(self, "_orig_fwd"):
            self.model.forward = self._orig_fwd
        self._patched = False

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        msg = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=prompt, images=[image], return_tensors="pt")
        inp = {k: v.to(self.model.device) for k, v in inp.items()}

        # 提取视觉特征
        pv = inp.get("pixel_values")
        vt = (self.model.model.vision_tower(pv).last_hidden_state
              if pv is not None and self.model.model.vision_tower
              else torch.zeros(1, 576, 1152, device=self.model.device, dtype=torch.float16))

        self._patch(vt)
        il = inp["input_ids"].shape[-1]
        out = self.model.generate(**inp, max_new_tokens=256, temperature=0.2, do_sample=False)
        if isinstance(out, tuple):
            out = out[0]
        ans = self.processor.decode(out[0][il:], skip_special_tokens=True)
        self._unpatch()
        return {"answer": ans, "num_passes": 2}
