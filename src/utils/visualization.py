"""追溯路径可视化 —— 展示 IVR 每次 pass 聚焦的区域"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Tuple
from pathlib import Path


def visualize_retracing(
    image_path: str,
    rois: List[Tuple[int, int, int, int]],
    pass_answers: List[str],
    pass_confidences: List[float],
    output_path: str = None,
):
    """生成迭代追溯路径可视化图

    Args:
        image_path: 原图路径
        rois: 每次 pass 的 ROI [(x1,y1,x2,y2), ...]
        pass_answers: 每次 pass 的输出文本
        pass_confidences: 每次 pass 的置信度
        output_path: 保存路径
    """
    n_passes = len(pass_answers)
    cols = min(n_passes + 1, 4)
    rows = (n_passes + 1 + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    img = Image.open(image_path)

    # Pass 1: 全图
    ax = axes[0]
    ax.imshow(img)
    ax.set_title(f"Pass 1 (全图)\nConf: {pass_confidences[0]:.2f}", fontsize=10)
    ax.axis("off")

    # Pass 2+: 聚焦区域
    for i in range(1, n_passes):
        if i < len(axes):
            ax = axes[i]
            ax.imshow(img)
            if i - 1 < len(rois):
                roi = rois[i - 1]
                rect = patches.Rectangle(
                    (roi[0], roi[1]),
                    roi[2] - roi[0],
                    roi[3] - roi[1],
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
            conf = pass_confidences[i] if i < len(pass_confidences) else 0
            ax.set_title(f"Pass {i+1} (聚焦)\nConf: {conf:.2f}", fontsize=10)
            ax.axis("off")

    # 关闭多余的 subplot
    for i in range(n_passes, len(axes)):
        axes[i].axis("off")

    plt.suptitle("IVR 迭代视觉追溯路径", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison_chart(
    baseline_names: List[str],
    chair_scores: List[float],
    pope_scores: List[float],
    avg_passes: List[int],
    output_path: str = None,
):
    """绘制 baseline 对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    # CHAIR
    axes[0].bar(baseline_names, [s * 100 for s in chair_scores], color=colors)
    axes[0].set_title("CHAIR (幻觉率 %)", fontsize=12)
    axes[0].set_ylabel("%")
    for i, v in enumerate(chair_scores):
        axes[0].text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center")

    # POPE
    axes[1].bar(baseline_names, [s * 100 for s in pope_scores], color=colors)
    axes[1].set_title("POPE Accuracy", fontsize=12)
    axes[1].set_ylabel("%")
    for i, v in enumerate(pope_scores):
        axes[1].text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center")

    # Avg Pass
    axes[2].bar(baseline_names, avg_passes, color=colors)
    axes[2].set_title("Average Passes", fontsize=12)
    for i, v in enumerate(avg_passes):
        axes[2].text(i, v + 0.1, str(v), ha="center")

    plt.suptitle("IVR vs Baselines 综合对比", fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
