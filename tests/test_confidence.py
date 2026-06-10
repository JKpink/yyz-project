"""测试置信度评估模块"""

import sys, os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ivr.confidence import ConfidenceEstimator


# ── 不需要 torch 的测试 ──

def test_should_stop_high_confidence():
    est = ConfidenceEstimator()
    assert est.should_retry(0.95, 0.9, 0.6) == "stop"


def test_should_refine_medium_confidence():
    est = ConfidenceEstimator()
    assert est.should_retry(0.75, 0.9, 0.6) == "refine"


def test_should_switch_low_confidence():
    est = ConfidenceEstimator()
    assert est.should_retry(0.4, 0.9, 0.6) == "switch"


# ── 需要 torch 的测试 ──

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_confidence_range():
    est = ConfidenceEstimator()
    fake_ids = torch.randint(0, 1000, (1, 10))
    conf = est.compute(fake_ids)
    assert 0.0 <= conf <= 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_length_penalty_short_answer():
    est = ConfidenceEstimator(length_min=5, length_penalty=0.1)
    fake_ids = torch.randint(0, 1000, (1, 2))
    conf = est.compute(fake_ids)
    assert conf < 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_logit_based_confidence():
    est = ConfidenceEstimator()
    fake_ids = torch.randint(0, 1000, (1, 15))
    logits = torch.zeros(1, 14, 100)
    logits[:, :, 42] = 10.0
    conf = est.compute(fake_ids, logits=logits, input_length=1)
    assert conf > 0.5
