"""测试配置加载模块"""

import sys, os, tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.config import Config


def test_config_loads_model():
    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "configs"))
    assert cfg.model_name == "openbmb/MiniCPM-V-4.6"
    assert cfg.device == "cuda"


def test_config_ivr_params():
    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "configs"))
    assert cfg.max_passes == 4
    assert cfg.confidence_high == 0.9
    assert cfg.confidence_low == 0.6
    assert 0 <= cfg.confidence_low < cfg.confidence_high <= 1.0
