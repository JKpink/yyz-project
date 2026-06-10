"""报告质量评测 —— ROUGE-L 等文本指标"""

from rouge_score import rouge_scorer
from typing import Dict, List
import numpy as np


class ReportMetrics:
    """评测生成的医学报告与参考报告的重叠度"""

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def compute_single(self, generated: str, reference: str) -> Dict:
        """单条评测"""
        scores = self.scorer.score(reference, generated)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    def compute_batch(
        self, generated_list: List[str], reference_list: List[str]
    ) -> Dict:
        """批量评测"""
        assert len(generated_list) == len(reference_list)

        all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

        for gen, ref in zip(generated_list, reference_list):
            scores = self.compute_single(gen, ref)
            for key in all_scores:
                all_scores[key].append(scores[key])

        return {
            "rouge1_mean": round(np.mean(all_scores["rouge1"]), 4),
            "rouge2_mean": round(np.mean(all_scores["rouge2"]), 4),
            "rougeL_mean": round(np.mean(all_scores["rougeL"]), 4),
            "rouge1_std": round(np.std(all_scores["rouge1"]), 4),
            "rougeL_std": round(np.std(all_scores["rougeL"]), 4),
            "count": len(generated_list),
        }
