"""测试 POPE 评测模块"""

import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluation.pope import POPEEvaluator


def test_all_correct():
    evaluator = POPEEvaluator()
    responses = [
        {"question": "Is there a car?", "pred_answer": "yes", "ground_truth": "yes"},
        {"question": "Is there a dog?", "pred_answer": "no", "ground_truth": "no"},
        {"question": "Is there a tree?", "pred_answer": "yes", "ground_truth": "yes"},
    ]
    result = evaluator.evaluate(responses)
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["hallucinations"] == 0  # fp = 0


def test_all_wrong():
    evaluator = POPEEvaluator()
    responses = [
        {"question": "Is there a car?", "pred_answer": "yes", "ground_truth": "no"},   # FP
        {"question": "Is there a dog?", "pred_answer": "yes", "ground_truth": "no"},    # FP
        {"question": "Is there a tree?", "pred_answer": "no", "ground_truth": "yes"},   # FN
    ]
    result = evaluator.evaluate(responses)
    assert result["accuracy"] == 0.0
    assert result["hallucinations"] == 2  # 2 FP


def test_chinese_yes_parsing():
    evaluator = POPEEvaluator()
    responses = [
        {"question": "图中有汽车吗？", "pred_answer": "是的，有一辆汽车", "ground_truth": "yes"},
        {"question": "图中有狗吗？", "pred_answer": "没有狗", "ground_truth": "no"},
    ]
    result = evaluator.evaluate(responses)
    assert result["accuracy"] == 1.0


def test_yes_ratio():
    evaluator = POPEEvaluator()
    responses = [
        {"question": "...", "pred_answer": "yes", "ground_truth": "yes"},
        {"question": "...", "pred_answer": "yes", "ground_truth": "no"},
        {"question": "...", "pred_answer": "no", "ground_truth": "no"},
    ]
    result = evaluator.evaluate(responses)
    assert result["yes_ratio"] == pytest.approx(round(2/3, 4))
