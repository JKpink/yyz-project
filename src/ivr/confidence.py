"""置信度评估模块 —— 判断模型对当前回答有多少把握"""

import torch
import torch.nn.functional as F
from typing import Optional


class ConfidenceEstimator:
    """评估模型生成结果的置信度

    使用两种策略的组合：
    1. Token-level: 生成序列的平均 log-probability
    2. Sequence-level: 答案长度是否合理（太短可能没想好）
    """

    def __init__(
        self,
        token_threshold: float = 0.9,
        length_min: int = 3,
        length_penalty: float = 0.05,
    ):
        self.token_threshold = token_threshold
        self.length_min = length_min
        self.length_penalty = length_penalty

    def compute(
        self,
        output_ids: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        input_length: int = 0,
    ) -> float:
        """
        Args:
            output_ids: 生成的 token IDs [batch, seq_len]
            logits: 每个生成步的 logits [batch, seq_len, vocab_size]
            input_length: 输入 token 数（用于截取生成部分）

        Returns:
            confidence ∈ [0, 1]
        """
        # 只评估生成的部分（非输入部分）
        if logits is not None:
            # token-level confidence from logits
            token_conf = self._token_confidence(logits, input_length)
        else:
            token_conf = 0.8  # 无 logits 时默认值

        # sequence-level confidence from length
        gen_length = output_ids.shape[-1] - input_length if input_length > 0 else output_ids.shape[-1]
        length_conf = self._length_confidence(gen_length)

        # 加权组合
        confidence = 0.8 * token_conf + 0.2 * length_conf
        return min(max(confidence, 0.0), 1.0)

    def _token_confidence(self, logits: torch.Tensor, input_length: int) -> float:
        """计算生成 token 的平均 softmax 概率"""
        gen_logits = logits[:, input_length - 1 : -1, :]  # 生成部分的 logits
        if gen_logits.shape[1] == 0:
            return 0.5
        probs = F.softmax(gen_logits, dim=-1)
        max_probs = probs.max(dim=-1).values  # [batch, gen_len]
        return max_probs.mean().item()

    def _length_confidence(self, gen_length: int) -> float:
        """越短的答案越没把握"""
        if gen_length >= self.length_min:
            return 1.0
        return 1.0 - self.length_penalty * (self.length_min - gen_length)

    def should_retry(
        self, confidence: float, threshold_high: float, threshold_low: float
    ) -> str:
        """
        Returns:
            "stop"  — 置信度够高，直接输出
            "refine" — 还不错，再看一次当前区域细化
            "switch" — 太低，换区域看
        """
        if confidence >= threshold_high:
            return "stop"
        elif confidence >= threshold_low:
            return "refine"
        else:
            return "switch"
