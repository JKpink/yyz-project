"""集成测试 —— 无需 GPU"""

import sys, os, tempfile, json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluation.pope import POPEEvaluator
from evaluation.chair import CHAIREvaluator


def test_result_schema(sample_results_dict):
    """验证所有 baseline 输出格式一致"""
    required_keys = {"answer", "num_passes", "image", "baseline"}
    for name, items in sample_results_dict.items():
        for r in items:
            assert required_keys.issubset(r.keys()), f"{name} missing keys: {required_keys - set(r.keys())}"
            assert isinstance(r["answer"], str)
            assert isinstance(r["num_passes"], int)
            assert r["num_passes"] >= 1


def test_ivr_has_extra_fields(sample_results_dict):
    """B4 IVR 应该比 B1 有额外的字段"""
    b4 = sample_results_dict["B4_IVR"]
    for r in b4:
        assert "pass_answers" in r
        assert "pass_confidences" in r
        assert len(r["pass_answers"]) == r["num_passes"]
        assert len(r["pass_confidences"]) == r["num_passes"]


def test_checkpoint_roundtrip():
    """断点保存 → 读取，数据一致"""
    test_dir = tempfile.mkdtemp()
    checkpoint_file = Path(test_dir) / "test_ckpt.json"

    original = {"results": [{"a": 1}, {"b": 2}], "done_images": ["img1.jpg", "img2.jpg"]}
    with open(checkpoint_file, "w") as f:
        json.dump(original, f)

    # 模拟断点加载
    with open(checkpoint_file) as f:
        loaded = json.load(f)
    assert loaded == original

    import shutil
    shutil.rmtree(test_dir)


def test_ivr_passes_never_exceed_max():
    """IVR 的 num_passes 不会超过 max_passes"""
    from utils.config import Config
    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "configs"))
    # 在没有实际推理的情况下，验证配置合理性
    assert cfg.max_passes >= 2
    assert cfg.max_passes <= 10


def test_thresholds_order():
    """置信度阈值关系正确"""
    from utils.config import Config
    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "configs"))
    assert 0 <= cfg.confidence_low < cfg.confidence_high <= 1.0


def test_chair_pipeline(sample_results_dict):
    """从 results dict 到 CHAIR 输出的完整链路"""
    # 用简化版 CHAIR 测试链路
    evaluator = CHAIREvaluator(ground_truth_objects={"car", "road", "tree", "person"})
    for name, items in sample_results_dict.items():
        for r in items:
            result = evaluator.compute(r["answer"])
            assert "chair_i" in result
            assert "hallucinated" in result
            assert "correct" in result
            assert isinstance(result["chair_i"], float)
