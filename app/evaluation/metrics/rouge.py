from typing import Any, Dict

from rouge_score import rouge_scorer

from app.evaluation.metrics.base_metric import BaseMetric


class ROUGEMetric(BaseMetric):
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores
    between a reference answer and model prediction.
    """

    def __init__(self):
        super().__init__(name="ROUGE")

        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ) -> Dict[str, Any]:

        if not reference or not prediction:
            return {
                "metric": self.name,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }

        scores = self.scorer.score(
            reference,
            prediction
        )

        return {
            "metric": self.name,
            "rouge1": round(
                scores["rouge1"].fmeasure,
                4
            ),
            "rouge2": round(
                scores["rouge2"].fmeasure,
                4
            ),
            "rougeL": round(
                scores["rougeL"].fmeasure,
                4
            )
        }