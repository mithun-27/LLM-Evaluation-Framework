from unittest.mock import patch

from app.evaluation.metrics.human_eval import (
    HumanEvaluationStore,
    SingleHumanEvaluationMetric,
)


@patch("builtins.input", side_effect=["5", "4", "5", "4", "Great answer."])
def test_single_human_evaluation(mock_input):
    metric = SingleHumanEvaluationMetric()
    res = metric.calculate(
        reference="Paris is the capital of France.",
        prediction="Paris is the capital of France.",
        prompt="What is the capital of France?"
    )

    assert res["metric"] == "Human Evaluation"
    assert res["correctness"] == 5
    assert res["relevance"] == 4
    assert res["fluency"] == 5
    assert res["helpfulness"] == 4
    assert res["average_score"] == 4.5
    assert res["comment"] == "Great answer."


def test_save_csv_human_evaluation(tmp_path):
    output_file = tmp_path / "human_evaluations.json"
    store = HumanEvaluationStore(output_path=str(output_file))

    results = [
        {
            "prompt": "Capital of France?",
            "reference": "Paris",
            "prediction": "Paris",
            "correctness": 5,
            "relevance": 5,
            "fluency": 5,
            "helpfulness": 5,
            "comment": "Accurate.",
            "average_score": 5.0
        }
    ]

    json_path = store.save(results)
    csv_path = store.save_csv(results)

    assert json_path.exists()
    assert csv_path.exists()
