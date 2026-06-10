"""POPE 评测 —— 判断模型中是否存在物体幻觉"""

from typing import Dict, List


class POPEEvaluator:
    """POPE (Polling-based Object Probing Evaluation)

    通过询问"图中是否有 X？"来评测 VLM 的物体幻觉。
    使用是/否判断准确率。

    参考: Li et al. 2023
    """

    def __init__(self):
        self.results = []

    def evaluate(
        self,
        responses: List[Dict],
    ) -> Dict:
        """
        Args:
            responses: [
                {
                    "question": "图中是否有汽车？",
                    "pred_answer": "yes" or "no",
                    "ground_truth": "yes" or "no",
                },
                ...
            ]

        Returns:
            {
                "accuracy": float,
                "precision": float,
                "recall": float,
                "f1": float,
                "yes_ratio": float,  # 模型说 yes 的比例
                "total": int,
                "correct": int,
            }
        """
        total = len(responses)
        correct = 0
        tp = fp = tn = fn = 0
        yes_count = 0

        for r in responses:
            pred = r["pred_answer"].lower().strip()
            gt = r["ground_truth"].lower().strip()

            # 归一化
            pred_yes = any(w in pred for w in ["yes", "是", "有", "存在", "可以"])
            gt_yes = any(w in gt for w in ["yes", "是", "有", "存在", "可以"])

            if pred_yes:
                yes_count += 1

            if pred_yes == gt_yes:
                correct += 1
                if pred_yes:
                    tp += 1
                else:
                    tn += 1
            else:
                if pred_yes:  # 说有但实际没有 → 幻觉
                    fp += 1
                else:        # 说没有但实际有 → 漏掉
                    fn += 1

        accuracy = correct / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        self.results = responses

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "yes_ratio": round(yes_count / total, 4) if total > 0 else 0.0,
            "total": total,
            "correct": correct,
            "hallucinations": fp,  # 假阳性 = 幻觉
        }
