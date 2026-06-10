"""配置解析器 —— 所有参数从 YAML 文件读取，代码不硬编码"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """统一的配置管理类"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs"

        self.config_dir = Path(config_dir)
        self._model = self._load("model.yaml")
        self._data = self._load("data.yaml")
        self._ivr = self._load("ivr.yaml")
        self._all = {**self._model, **self._data, **self._ivr}

    def _load(self, filename: str) -> Dict[str, Any]:
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ── 便捷属性 ──────────────────────────────

    @property
    def model(self) -> dict:
        return self._model["model"]

    @property
    def inference(self) -> dict:
        return self._model["inference"]

    @property
    def data(self) -> dict:
        return self._data

    @property
    def ivr(self) -> dict:
        return self._ivr["ivr"]

    @property
    def memvr(self) -> dict:
        return self._ivr.get("memvr", {})

    @property
    def all(self) -> dict:
        return self._all

    # ── 模型 ──

    @property
    def model_name(self) -> str:
        return self.model["name"]

    @property
    def device(self) -> str:
        return self.model.get("device", "cuda")

    # ── IVR ──

    @property
    def max_passes(self) -> int:
        return self.ivr["max_passes"]

    @property
    def confidence_high(self) -> float:
        return self.ivr["confidence_threshold_high"]

    @property
    def confidence_low(self) -> float:
        return self.ivr["confidence_threshold_low"]


# 全局单例
_config_instance = None


def get_config(config_dir: str = None) -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_dir)
    return _config_instance
