"""共享 fixtures —— 无需 GPU"""

import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# torch 是可选的（本地无 GPU 时不装）
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def fake_visual_features():
    """模拟 CLIP 视觉特征 [B=1, N=576, D=1024]"""
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    return torch.randn(1, 576, 1024)


@pytest.fixture
def fake_output_ids():
    """模拟生成的 token IDs [B=1, seq_len=15]"""
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    return torch.randint(0, 1000, (1, 15))


@pytest.fixture
def fake_high_conf_logits():
    """模拟高置信度 logits: 每步对 token 42 概率极高"""
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    logits = torch.randn(1, 14, 100)
    logits[:, :, 42] = 10.0
    return logits


@pytest.fixture
def sample_gt_objects():
    """样例 ground truth 物体集合"""
    return {"car", "person", "tree", "road"}


@pytest.fixture
def sample_results_dict():
    """模拟 4 个 baseline 的结果 dict"""
    return {
        "B1_Direct": [
            {"image": "img001.jpg", "answer": "A car on the road.", "num_passes": 1, "baseline": "B1_Direct"},
            {"image": "img002.jpg", "answer": "A person and a tree.", "num_passes": 1, "baseline": "B1_Direct"},
        ],
        "B2_Thinking": [
            {"image": "img001.jpg", "answer": "I see a car and a road.", "num_passes": 1, "baseline": "B2_Thinking"},
            {"image": "img002.jpg", "answer": "A person next to a tree.", "num_passes": 1, "baseline": "B2_Thinking"},
        ],
        "B3_MemVR": [
            {"image": "img001.jpg", "answer": "Car on road.", "num_passes": 2, "baseline": "B3_MemVR"},
            {"image": "img002.jpg", "answer": "Person and tree.", "num_passes": 2, "baseline": "B3_MemVR"},
        ],
        "B4_IVR": [
            {"image": "img001.jpg", "answer": "A car driving on the road.", "num_passes": 3, "baseline": "B4_IVR", "pass_answers": ["...", "...", "A car..."], "pass_confidences": [0.5, 0.7, 0.92]},
            {"image": "img002.jpg", "answer": "A person standing by a tree.", "num_passes": 4, "baseline": "B4_IVR", "pass_answers": ["...", "...", "...", "A person..."], "pass_confidences": [0.4, 0.6, 0.75, 0.91]},
        ],
    }
