"""B3: MemVR — 内置 monkey-patch 实现视觉追溯

Monkey-patch model.forward: 逐层监控 token 熵值, 超标时注入视觉特征到 hidden state
架构: model.model.language_model.layers[i].mlp (Qwen3_5MLP, 24层)
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

    def _patch(self):
        if self._patched: return
        self._patched = True
        patch = self
        layers = self.model.model.language_model.layers
        N = len(layers)
        lm_head = self.model.lm_head

        # 1. 给每个 MLP 加 MemVR 属性
        for i in range(N):
            mlp = layers[i].mlp
            mlp._orig = mlp.forward
            mlp._active = patch.starting_layer <= i <= patch.ending_layer
            mlp._sign = 0
            mlp._ratio = patch.retracing_ratio
            mlp._w1 = None
            mlp._w2 = None

        # 2. 替换 MLP forward（加适配通道）
        def make_fwd(m):
            o = m._orig
            def f(x):
                dtype = next(m.parameters()).dtype
                x = x.to(dtype)
                out = o(x)
                if m._sign:
                    D = x.shape[-1]
                    if m._w1 is None:
                        m._w1 = torch.nn.Linear(D, D, bias=False).to(device=x.device, dtype=dtype)
                        m._w2 = torch.nn.Linear(D, D, bias=False).to(device=x.device, dtype=dtype)
                    a = m._w2(torch.relu(m._w1(x))).to(dtype)
                    s = out.abs().mean() / (a.abs().mean() + 1e-8)
                    out = out * (1 - m._ratio) + a * s * m._ratio
                    m._sign = 0
                return out
            return f

        for i in range(N):
            layers[i].mlp.forward = make_fwd(layers[i].mlp)

        # 3. 替换 model.forward（加逐层熵监控）
        patch._orig_fwd = self.model.forward

        def new_fwd(self_m, input_ids=None, attention_mask=None, position_ids=None,
                     past_key_values=None, inputs_embeds=None, pixel_values=None,
                     image_sizes=None, use_cache=None, output_attentions=None,
                     output_hidden_states=None, return_dict=None, cache_position=None,
                     downsample_mode=None, **kw):
            out = patch._orig_fwd(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                image_sizes=image_sizes, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True, return_dict=True,
                cache_position=cache_position, downsample_mode=downsample_mode, **kw)
            hs = out.get("hidden_states")
            if not hs: return out
            done = False
            ll = self_m.model.language_model.layers
            for idx in range(N):
                t = hs[idx][0, -1, :].float()
                lg = lm_head(t)
                p = F.softmax(torch.topk(lg, 10).values.float(), dim=-1)
                e = (-p * torch.log(p + 1e-10)).sum().item() / np.log(10)
                if not done and patch.starting_layer <= idx <= patch.ending_layer and e > patch.entropy_threshold:
                    done = True
                    nx = min(idx + 1, N - 1)
                    mlp = ll[nx].mlp
                    if mlp._active: mlp._sign = 1
            return out

        self.model.forward = new_fwd.__get__(self.model, type(self.model))

    def _unpatch(self):
        if not self._patched: return
        for m in self.model.model.language_model.layers:
            if hasattr(m.mlp, "_orig"): m.mlp.forward = m.mlp._orig
        if hasattr(self, "_orig_fwd"): self.model.forward = self._orig_fwd
        self._patched = False

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        msg = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}]}]
        inputs = self.processor.apply_chat_template(
            msg, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            downsample_mode="16x",
        ).to(self.model.device)
        il = inputs.input_ids.shape[-1]

        self._patch()
        out = self.model.generate(**inputs, downsample_mode="16x", max_new_tokens=256)
        if isinstance(out, tuple): out = out[0]
        ans = self.processor.decode(out[0][il:], skip_special_tokens=True)
        self._unpatch()
        return {"answer": ans, "num_passes": 2}
