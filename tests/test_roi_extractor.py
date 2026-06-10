"""测试 ROI 提取模块"""

import sys, os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ivr.roi_extractor import ROIExtractor


def make_features():
    return torch.randn(1, 576, 1024)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_roi_grid_strategy():
    extractor = ROIExtractor(strategy="grid", num_rois=2)
    rois = extractor.extract(make_features(), "test", image_size=(336, 336))
    assert len(rois) == 2
    for (x1, y1, x2, y2) in rois:
        assert 0 <= x1 < x2 <= 336
        assert 0 <= y1 < y2 <= 336


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_roi_activation_strategy():
    extractor = ROIExtractor(strategy="attention", num_rois=3)
    rois = extractor.extract(make_features(), "test", image_size=(336, 336))
    assert len(rois) <= 3
    for (x1, y1, x2, y2) in rois:
        assert 0 <= x1 < x2 <= 336
        assert 0 <= y1 < y2 <= 336


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_hybrid_strategy():
    extractor = ROIExtractor(strategy="hybrid", num_rois=4)
    rois = extractor.extract(make_features(), "test", image_size=(336, 336))
    assert len(rois) <= 4
