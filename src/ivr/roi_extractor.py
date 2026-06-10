"""ROI 提取模块 —— 从模型输出中定位值得再看一遍的图像区域"""

import torch
from typing import List, Tuple, Optional


class ROIExtractor:
    """提取感兴趣区域（Region of Interest）

    支持三种策略：
    - attention: 基于注意力权重定位
    - grid: 均匀网格划分
    - hybrid: 先 attention 定位，不足时用 grid 补充
    """

    def __init__(self, strategy: str = "attention", num_rois: int = 3):
        self.strategy = strategy
        self.num_rois = num_rois

    def extract(
        self,
        visual_features: torch.Tensor,
        answer_text: str,
        attentions: Optional[List[torch.Tensor]] = None,
        image_size: Tuple[int, int] = (336, 336),
    ) -> List[Tuple[int, int, int, int]]:
        """
        Args:
            visual_features: [B, N_v, D] 视觉特征
            answer_text: 当前 pass 的文本输出
            attentions: 各层的注意力权重（可选）
            image_size: 原图尺寸 (H, W)

        Returns:
            List of (x1, y1, x2, y2) bboxes
        """
        # 默认用 grid（不依赖模型内部状态）
        if self.strategy == "attention" and attentions is not None:
            return self._extract_by_attention(attentions, image_size)
        elif self.strategy == "activation" and visual_features is not None:
            return self._extract_by_activation(visual_features, image_size)
        else:
            # 降级到 grid（最可靠，不需要模型内部信息）
            return self._extract_by_grid(image_size)

    def _extract_by_attention(
        self,
        attentions: List[torch.Tensor],
        image_size: Tuple[int, int],
    ) -> List[Tuple[int, int, int, int]]:
        """基于最后层注意力定位 ROI"""
        H, W = image_size

        # 取最后几层的平均注意力
        last_attns = attentions[-4:] if len(attentions) >= 4 else attentions
        avg_attn = torch.stack(last_attns).mean(dim=0)  # [B, heads, seq, seq]

        # 对视觉 token 的注意力求和
        num_visual = 576  # CLIP: (336/14)^2
        visual_attn = avg_attn[:, :, :, :num_visual].mean(dim=(0, 1, 2))  # [num_visual]

        # 重塑为空间网格 24×24
        grid_size = 24
        attn_map = visual_attn.reshape(grid_size, grid_size)

        # 找 top-k 最受关注的 patch
        patch_size = W // grid_size
        flat = attn_map.flatten()
        _, top_indices = torch.topk(flat, self.num_rois)

        rois = []
        for idx in top_indices:
            row, col = idx.item() // grid_size, idx.item() % grid_size
            # 以该 patch 为中心扩展 3×3 区域
            r_start = max(0, row - 1)
            r_end = min(grid_size, row + 2)
            c_start = max(0, col - 1)
            c_end = min(grid_size, col + 2)
            rois.append((
                c_start * patch_size,
                r_start * patch_size,
                c_end * patch_size,
                r_end * patch_size,
            ))
        return rois

    def _extract_by_grid(
        self, image_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """均匀网格划分 ROI"""
        H, W = image_size
        rows, cols = 2, 2  # 将图分 2×2=4 块
        cell_h, cell_w = H // rows, W // cols

        rois = []
        for r in range(rows):
            for c in range(cols):
                rois.append((
                    c * cell_w,
                    r * cell_h,
                    (c + 1) * cell_w,
                    (r + 1) * cell_h,
                ))
        return rois[: self.num_rois]

    def _extract_by_activation(
        self,
        visual_features: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> List[Tuple[int, int, int, int]]:
        """基于视觉特征 L2 范数定位活跃区域"""
        H, W = image_size
        N_v = visual_features.shape[1]
        grid_size = int(N_v**0.5)

        # 计算每个 patch 的激活强度
        activation = visual_features.norm(dim=-1).squeeze(0)  # [N_v]
        act_map = activation.reshape(grid_size, grid_size)

        patch_size = W // grid_size
        flat = act_map.flatten()
        _, top_indices = torch.topk(flat, self.num_rois)

        rois = []
        for idx in top_indices:
            row, col = idx.item() // grid_size, idx.item() % grid_size
            r_start = max(0, row - 1)
            r_end = min(grid_size, row + 2)
            c_start = max(0, col - 1)
            c_end = min(grid_size, col + 2)
            rois.append((
                c_start * patch_size,
                r_start * patch_size,
                c_end * patch_size,
                r_end * patch_size,
            ))
        return rois
