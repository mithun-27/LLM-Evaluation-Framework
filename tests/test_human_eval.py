import json
from unittest.mock import patch

from app.evaluation.metrics.human_eval import (
    HumanEvaluationMetric,
    HumanEvaluationStore,
)


@patch(
    "builtins.input",
    side_effect=["1", "Model A is more accurate."]
)
def test_human_evaluation_model_a(mock_input):
    metric = HumanEvaluationMetric()

    result = metric.calculate(
        reference="Paris is the capital of France.",
        prediction="Paris is the capital of France.",
        prompt="What is the capital of France?",
        model_a_answer=(
            "Paris is the capital of France."
        ),
        model_b_answer="Paris is a city in Europe."
    )

    print("\nHuman Evaluation Result:")
    print(result)

    assert result["winner"] == "Model A"
    assert result["comment"] == (
        "Model A is more accurate."
    )


@patch(
    "builtins.input",
    side_effect=["3", ""]
)
def test_human_evaluation_tie(mock_input):
    metric = HumanEvaluationMetric()

    result = metric.calculate(
        reference="Jupiter is the largest planet.",
        prediction="Jupiter is the largest planet.",
        prompt="What is the largest planet?",
        model_a_answer="Jupiter is the largest planet.",
        model_b_answer="The largest planet is Jupiter."
    )

    assert result["winner"] == "Tie"


def test_save_human_evaluation(tmp_path):
    output_file = (
        tmp_path / "human_evaluations.json"
    )

    store = HumanEvaluationStore(
        output_path=str(output_file)
    )

    results = [
        {
            "winner": "Model A",
            "comment": "Better answer"
        }
    ]

    saved_path = store.save(results)

    assert saved_path.exists()

    with saved_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        report = json.load(file)

    assert report["total_evaluations"] == 1
    assert len(report["evaluations"]) == 1