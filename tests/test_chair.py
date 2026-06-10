"""测试 CHAIR 评测模块"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluation.chair import CHAIREvaluator


def test_chair_all_correct():
    evaluator = CHAIREvaluator(ground_truth_objects={"car", "person", "tree"})
    result = evaluator.compute(
        caption="A car and a person near a tree.",
        objects_in_caption=["car", "person", "tree"],
    )
    assert result["hallucinated"] == []
    assert result["correct"] == ["car", "person", "tree"]
    assert result["chair_i"] == 0.0


def test_chair_all_hallucinated():
    evaluator = CHAIREvaluator(ground_truth_objects={"car", "road"})
    result = evaluator.compute(
        caption="A dog and a cat on a boat.",
        objects_in_caption=["dog", "cat", "boat"],
    )
    assert len(result["hallucinated"]) == 3
    assert result["chair_i"] == 1.0


def test_chair_partial():
    evaluator = CHAIREvaluator(ground_truth_objects={"car", "person", "tree"})
    result = evaluator.compute(
        caption="A car and a dog.",
        objects_in_caption=["car", "dog"],
    )
    assert "dog" in result["hallucinated"]
    assert "car" in result["correct"]
    assert result["chair_i"] == 0.5
    assert result["total_objects"] == 2


def test_auto_extract_objects():
    evaluator = CHAIREvaluator(ground_truth_objects={"lung", "heart", "nodule"})
    result = evaluator.compute("The lung shows a nodule but heart is normal.")
    assert len(result["hallucinated"]) + len(result["correct"]) == result["total_objects"]
