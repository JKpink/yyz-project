"""CHAIR 评测 —— 评估 VLM 描述中的幻觉率"""

import re
from typing import List, Set


class CHAIREvaluator:
    """评估图像描述中的物体幻觉

    CHAIR (Caption Hallucination Assessment with Image Object References)
    测量模型生成的描述中有多少比例的对象实际上不存在于图中。

    参考: Rohrbach et al. 2018
    """

    def __init__(self, ground_truth_objects: Set[str] = None):
        """
        Args:
            ground_truth_objects: 图中真实存在的物体名称集合
        """
        self.gt_objects = ground_truth_objects or set()

    def set_ground_truth(self, objects: Set[str]):
        """设置当前图像的真实物体"""
        self.gt_objects = {o.lower() for o in objects}

    def compute(self, caption: str, objects_in_caption: List[str] = None) -> dict:
        """
        Args:
            caption: 模型生成的描述
            objects_in_caption: 从描述中提取的对象（若为 None 则自动提取名词）

        Returns:
            {
                "chair_s": float,      # sentence-level CHAIR
                "chair_i": float,      # instance-level CHAIR
                "hallucinated": [str], # 幻觉对象列表
                "correct": [str],      # 正确识别的对象
                "total_objects": int,  # 描述中总对象数
            }
        """
        if objects_in_caption is None:
            objects_in_caption = self._extract_objects(caption)

        objects_in_caption = [o.lower() for o in objects_in_caption]

        hallucinated = []
        correct = []
        for obj in objects_in_caption:
            if obj in self.gt_objects:
                correct.append(obj)
            else:
                # 检查是否为 ground truth 中子串
                matched = False
                for gt_obj in self.gt_objects:
                    if obj in gt_obj or gt_obj in obj:
                        correct.append(obj)
                        matched = True
                        break
                if not matched:
                    hallucinated.append(obj)

        total = len(objects_in_caption)
        chair_i = len(hallucinated) / total if total > 0 else 0.0
        chair_s = 1.0 if hallucinated else 0.0

        return {
            "chair_i": round(chair_i, 4),
            "chair_s": chair_s,
            "hallucinated": hallucinated,
            "correct": correct,
            "total_objects": total,
        }

    def _extract_objects(self, caption: str) -> List[str]:
        """从描述中提取名词短语（简单实现）"""
        # 移除标点
        text = re.sub(r"[^\w\s]", " ", caption.lower())
        words = text.split()

        # 常见物体关键词匹配（简化版）
        common_objects = {
            "lung", "heart", "nodule", "lesion", "fracture", "bone",
            "vessel", "artery", "vein", "tumor", "mass", "cavity",
            "effusion", "pneumonia", "infiltrate", "opacity", "shadow",
            "catheter", "tube", "line", "clip", "stent", "valve",
            "rib", "spine", "diaphragm", "pleura", "mediastinum",
            # 通用医学
            "patient", "doctor", "x-ray", "image", "scan",
        }

        found = set()
        for word in words:
            if word in common_objects:
                found.add(word)
            # 检查 bigram
            for i, w1 in enumerate(words[:-1]):
                bigram = f"{w1} {words[i+1]}"
                if bigram in common_objects:
                    found.add(bigram)

        return list(found)
