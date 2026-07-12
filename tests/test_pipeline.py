import pandas as pd

from app.evaluation.evaluator import Evaluator
from app.evaluation.metrics.base_metric import BaseMetric


class DummyMetric(BaseMetric):

    def __init__(self):
        super().__init__(name="Dummy Metric")

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ):
        score = 1.0 if reference == prediction else 0.0

        return {
            "metric": self.name,
            "score": score
        }


def test_evaluator_pipeline():
    metric = DummyMetric()
    evaluator = Evaluator([metric])

    df = pd.DataFrame([
        {
            "prompt": "Capital of France?",
            "reference_answer": "Paris",
            "model_answer": "Paris",
            "context": ""
        },
        {
            "prompt": "Capital of Germany?",
            "reference_answer": "Berlin",
            "model_answer": "Munich",
            "context": ""
        }
    ])

    results = evaluator.evaluate_dataset(df)

    assert "summary" in results
    assert "results" in results

    summary = results["summary"]
    assert summary["total_rows"] == 2
    assert summary["overall_score"] == 50.0  # 1.0 and 0.0 scores averaged

    avg_score = summary["metrics"]["Dummy Metric"]["score"]
    assert avg_score == 0.5