#!/usr/bin/env python3
"""主评测脚本 —— 运行所有 baseline 并输出对比结果

用法:
    python src/evaluate.py --config configs/ivr.yaml --data data/IU-X-Ray --max-images 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_config
from ivr import IVRInference, ConfidenceEstimator, ROIExtractor
from baselines.baseline_direct import BaselineDirect
from baselines.baseline_cot import BaselineCoT
from baselines.baseline_memvr import BaselineMemVR
from evaluation.chair import CHAIREvaluator
from evaluation.pope import POPEEvaluator


def load_model_and_processor(config):
    """加载 MiniCPM-V-4.6-int4"""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_name = config.model_name
    print(f"[Loading] {model_name}")

    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )
    if config.model.get("use_quantized"):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


def run_baseline(baseline, images, questions, name: str):
    """运行单个 baseline 并计时"""
    print(f"\n{'='*50}")
    print(f"[Running] {name}")
    print(f"{'='*50}")

    results = []
    start_time = time.time()

    for i, (img, q) in enumerate(zip(images, questions)):
        result = baseline.generate(img, q)
        result["baseline"] = name
        result["image_idx"] = i
        results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{name}] {i+1}/{len(images)} | {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"  [{name}] Done. {len(images)} images in {elapsed:.1f}s "
          f"({elapsed/len(images):.2f}s/img)")

    return results, elapsed


def main():
    parser = argparse.ArgumentParser(description="IVR Evaluation Pipeline")
    parser.add_argument("--config", default="configs/ivr.yaml",
                        help="Path to config directory")
    parser.add_argument("--data", default="data/IU-X-Ray",
                        help="Path to image directory")
    parser.add_argument("--max-images", type=int, default=100,
                        help="Max images to evaluate")
    parser.add_argument("--output", default="results/baselines.json",
                        help="Output JSON path")
    parser.add_argument("--baseline", default="all",
                        choices=["all", "direct", "cot", "memvr", "ivr"],
                        help="Which baseline to run")
    args = parser.parse_args()

    # 加载配置
    config = get_config(Path(args.config).parent
                        if Path(args.config).is_file()
                        else args.config)

    # 加载模型
    model, processor = load_model_and_processor(config)

    # 加载数据（简化：从文件夹读取图像）
    from PIL import Image
    data_dir = Path(args.data)
    image_files = sorted(data_dir.glob("*.png")) + sorted(data_dir.glob("*.jpg"))
    image_files = image_files[: args.max_images]

    images = [Image.open(f).convert("RGB") for f in image_files]
    # 医学场景的标准问题
    questions = ["请描述这张胸片中的异常发现。如果未发现异常，请说明是正常胸片。"
                 for _ in images]

    print(f"\n[Data] {len(images)} images loaded")

    all_results = {}
    timing = {}

    # ── B1: 直接回答 ──
    if args.baseline in ("all", "direct"):
        b1 = BaselineDirect(model, processor)
        results, elapsed = run_baseline(b1, images, questions, "B1_Direct")
        all_results["B1_Direct"] = results
        timing["B1_Direct"] = round(elapsed, 1)

    # ── B2: CoT ──
    if args.baseline in ("all", "cot"):
        b2 = BaselineCoT(model, processor)
        results, elapsed = run_baseline(b2, images, questions, "B2_CoT")
        all_results["B2_CoT"] = results
        timing["B2_CoT"] = round(elapsed, 1)

    # ── B3: MemVR ──
    if args.baseline in ("all", "memvr"):
        b3 = BaselineMemVR(model, processor, retrace_layers=[16, 20, 23])
        results, elapsed = run_baseline(b3, images, questions, "B3_MemVR")
        all_results["B3_MemVR"] = results
        timing["B3_MemVR"] = round(elapsed, 1)

    # ── B4: IVR ──
    if args.baseline in ("all", "ivr"):
        b4 = IVRInference(model, processor, config)
        results, elapsed = run_baseline(b4, images, questions, "B4_IVR")
        all_results["B4_IVR"] = results
        timing["B4_IVR"] = round(elapsed, 1)

    # ── 汇总 ──
    summary = {
        "config": {
            "model": config.model_name,
            "num_images": len(images),
            "max_passes": config.max_passes,
        },
        "timing_seconds": timing,
        "passes": {},
    }

    for name, results in all_results.items():
        avg_passes = sum(r.get("num_passes", 1) for r in results) / len(results)
        summary["passes"][name] = round(avg_passes, 2)
        print(f"  {name}: avg_passes={avg_passes:.2f}")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[Saved] Results → {output_path}")


if __name__ == "__main__":
    main()
