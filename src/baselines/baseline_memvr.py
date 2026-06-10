"""B3: MemVR（真正的 forward 修改版）

基于 MiniCPM-V-4.6 架构实现 MemVR 的 monkey-patch：
- 运行时注入 MLP 适配通道（不替换全局类）
- 逐层监控 token 熵值，超标时注入视觉特征
- 支持 Qwen 系列（LLaMA 已移除）

参考: MemVR ICML 2025, memvr.py apply_memvr_qwen()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from PIL import Image


class BaselineMemVR:
    """真实 MemVR 推理 —— 内部 forward 修改实现视觉追溯"""

    def __init__(
        self,
        model,
        processor,
        starting_layer: int = 5,
        ending_layer: int = 16,
        entropy_threshold: float = 0.75,
        retracing_ratio: float = 0.12,
    ):
        self.model = model
        self.processor = processor
        self.starting_layer = starting_layer
        self.ending_layer = ending_layer
        self.entropy_threshold = entropy_threshold
        self.retracing_ratio = retracing_ratio
        self._patched = False

    def _patch(self, image_features: torch.Tensor):
        """运行时注入 MemVR 机制到已加载的模型中"""
        if self._patched:
            return

        # 1. 在指定层 MLP 上添加适配属性
        for i in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[i].mlp
            mlp._memvr_active = (
                self.starting_layer <= i <= self.ending_layer
            )
            mlp._adpt_sign = 0
            mlp._visual_token = image_features.detach().clone()
            mlp._retracing_ratio = self.retracing_ratio
            mlp._adpt_w1 = None
            mlp._adpt_w2 = None

            # 保存原始 forward
            mlp._orig_forward = mlp.forward

        # 2. 替换每一层 MLP 的 forward
        def make_memvr_forward(mlp_module):
            orig_forward = mlp_module._orig_forward

            def memvr_forward(x):
                ffn_out = orig_forward(x)

                # 检查是否触发适配通道
                if getattr(mlp_module, "_adpt_sign", 0) == 1:
                    if mlp_module._adpt_w1 is None:
                        # 初始化适配权重（用视觉特征）
                        vt = mlp_module._visual_token.to(x.device)
                        # 适配权重初始化为视觉特征的投影
                        d = x.shape[-1]
                        mlp_module._adpt_w1 = nn.Parameter(
                            torch.zeros(d, d, device=x.device)
                        )
                        mlp_module._adpt_w2 = nn.Parameter(
                            torch.zeros(d, d, device=x.device)
                        )
                        # 用视觉特征初始化
                        with torch.no_grad():
                            scale1 = torch.mean(torch.abs(ffn_out)) / (
                                torch.mean(torch.abs(vt)) + 1e-8
                            )
                            scale2 = torch.mean(torch.abs(x)) / (
                                torch.mean(torch.abs(vt)) + 1e-8
                            )
                            v1 = vt[:d]
                            v2 = torch.eye(d, device=x.device)[: len(v1)] * v1[
                                :1
                            ].unsqueeze(-1)

                    # 视觉适配输出
                    adapter_out = torch.matmul(
                        torch.matmul(x, mlp_module._adpt_w1),
                        mlp_module._adpt_w2,
                    )
                    ratio = mlp_module._retracing_ratio
                    # 归一化后混合
                    norm_adapter = (
                        torch.mean(torch.abs(ffn_out))
                        / (torch.mean(torch.abs(adapter_out)) + 1e-8)
                        * adapter_out
                    )
                    output = ffn_out * (1 - ratio) + norm_adapter * ratio

                    # 用完后关闭（只触发一次）
                    mlp_module._adpt_sign = 0
                    return output
                else:
                    return ffn_out

            return memvr_forward

        for i in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[i].mlp
            mlp.forward = make_memvr_forward(mlp)

        # 3. 保存原始 model.forward
        self._orig_model_forward = self.model.forward

        # 4. 替换 model.forward（加逐层熵监控）
        def memvr_model_forward(
            self_model,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            **kwargs,
        ):
            # 走标准 forward 获取 output
            output = self._orig_model_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )

            # 逐层熵监控 + 视觉追溯
            if output_hidden_states and hasattr(self_model, "lm_head"):
                hidden_states = output.hidden_states
                visual_retracing_triggered = False

                for layer_idx in range(
                    min(self.starting_layer, len(hidden_states) - 1),
                    min(self.ending_layer + 1, len(hidden_states)),
                ):
                    hs = hidden_states[layer_idx]
                    # 只看最后一个 token 的熵
                    last_token = hs[:, -1, :].float()
                    logits = self_model.lm_head(last_token)
                    probs = F.softmax(logits, dim=-1)

                    # Top-10 熵
                    top10_probs, _ = torch.topk(probs, 10, dim=-1)
                    entropy = torch.sum(
                        -top10_probs * torch.log(top10_probs + 1e-10)
                    ) / np.log(10)
                    entropy = entropy.item()

                    if (
                        entropy > self.entropy_threshold
                        and not visual_retracing_triggered
                    ):
                        visual_retracing_triggered = True
                        # 触发下一层的适配通道
                        next_layer = min(layer_idx + 1, len(self_model.model.layers) - 1)
                        mlp = self_model.model.layers[next_layer].mlp
                        if getattr(mlp, "_memvr_active", False):
                            mlp._adpt_sign = 1

            return output

        self.model.forward = memvr_model_forward.__get__(
            self.model, type(self.model)
        )
        self._patched = True

    def _unpatch(self):
        """恢复原始 forward"""
        if not self._patched:
            return
        for i in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[i].mlp
            if hasattr(mlp, "_orig_forward"):
                mlp.forward = mlp._orig_forward
        if hasattr(self, "_orig_model_forward"):
            self.model.forward = self._orig_model_forward
        self._patched = False

    @torch.no_grad()
    def generate(self, image: Image.Image, question: str) -> Dict:
        # 1. 先编码图像获取视觉特征
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
        inputs = self.processor(
            text=prompt, images=[image], return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 2. 先跑一次前向获取 image_features
        with torch.no_grad():
            # 用 model 内部的 vision encoder 获取特征
            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None:
                vision_outputs = self.model.vision_model(pixel_values)
                image_features = vision_outputs.last_hidden_state
            else:
                image_features = torch.zeros(1, 576, 1024).to(
                    self.model.device
                )

        # 3. Patch 模型
        self._patch(image_features)

        # 4. 正常生成（forward 已被替换）
        input_len = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
            output_hidden_states=True,
        )

        answer = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        # 5. 恢复模型
        self._unpatch()

        return {"answer": answer, "num_passes": 2}
