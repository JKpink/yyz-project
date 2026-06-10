# IVR: Iterative Visual Retracing

**基于迭代视觉追溯的 VLM 幻觉缓解方法 —— 以医学影像诊断为例**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Transformers](https://img.shields.io/badge/Transformers-≥5.7.0-orange)](https://huggingface.co/docs/transformers)
[![GPU](https://img.shields.io/badge/GPU-T4%2016GB-green)]()

---

## 一句话概括

让 VLM "多看几遍图"，每次聚焦不同区域，而不是"多想几句"——证明迭代回看比 CoT 更有效减少幻觉。

---

## 核心思路

```
B1 (直接回答):    [看图] → 答案                          ❌ 看漏了
B2 (CoT):         [看图] → "我想想..." → 答案             ❌ 没真再看
B3 (MemVR):       [看图] → [回看记忆] → 答案              ⚠️ 只一次
B4 (IVR, 我们的): [全图] → [聚焦ROI₁] → [聚焦ROI₂] → 答案  ✅ 迭代验证
```

---

## 项目结构

```
yyz-project/
├── src/
│   ├── ivr/                  # IVR 核心：推理引擎、ROI 提取、置信度评估
│   ├── baselines/            # B1/B2/B3 三个 baseline 实现
│   ├── evaluation/           # CHAIR、POPE、ROUGE-L 评测
│   ├── utils/                # 配置解析、可视化
│   └── evaluate.py           # 主评测入口
├── configs/                  # 参数与代码分离（YAML）
├── repositories/MemVR/       # MemVR 参考实现（ICML 2025, 173⭐）
└── data/IU-X-Ray/            # 胸片数据集（需自行下载）
```

---

## 快速开始

### 1. 环境

```bash
pip install "transformers[torch]>=5.7.0" torchvision av pyyaml rouge-score matplotlib Pillow
```

### 2. 加载模型

模型通过 HuggingFace 自动缓存，无需手动下载：

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "openbmb/MiniCPM-V-4.6",
    trust_remote_code=True,
    device_map="auto",
    load_in_4bit=True,  # int4 量化，省一半显存
)
processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-4.6")
```

首次自动下载 ~1.5GB（int4 量化），后续读缓存。

### 3. 运行评测

```bash
# 单 baseline
python src/evaluate.py --baseline ivr --max-images 100

# 全部对比
python src/evaluate.py --baseline all --max-images 500 --output results/baselines.json
```

---

## Baseline

| # | 名称 | 说明 | Pass 数 |
|---|------|------|---------|
| B1 | 直接回答 | 标准 VLM 推理 | 1 |
| B2 | CoT | Prompt 加"一步步思考" | 1（软推理） |
| B3 | MemVR（移植版） | 全图→回看记忆（参考 ICML 2025） | 2 |
| B4 | **IVR（本方法）** | 全图→迭代聚焦→验证 | 2-4（自适应） |

---

## 评测指标

| 指标 | 含义 |
|------|------|
| CHAIR | 描述中出现图中不存在物体的比例（越低越好） |
| POPE Accuracy | 判断"图中是否有X"的准确率 |
| ROUGE-L | 与参考报告的重叠度 |
| Avg Pass | 平均追溯次数（效率） |

---

## 算力需求

| 项目 | 值 |
|------|-----|
| 模型 | MiniCPM-V-4.6 (1.3B, load_in_4bit) |
| 推理显存 | ~10 GB（IVR 全模式） |
| 最低 GPU | 单 T4 16GB |
| 评测时间 (7500 张) | ~2-3 小时 |
| 训练 | 无（纯推理时机制） |

---

## 关键参考

| 工作 | 会议 | Stars | 用途 |
|------|------|-------|------|
| [MemVR](https://github.com/1zhou-Wang/MemVR) | ICML 2025 | ⭐173 | 追溯机制 baseline |
| [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) | — | ⭐25k | 基座模型 |
| [POPE](https://github.com/RUCAIBox/POPE) | — | ⭐262 | 幻觉评测 |

---

## License

Apache 2.0
