from bert_score import BERTScorer

from app.evaluation.metrics.base_metric import BaseMetric


class BERTScoreMetric(BaseMetric):
    """
    Calculates semantic similarity using BERTScore.
    """

    def __init__(self):
        super().__init__(name="BERTScore")

        self.scorer = BERTScorer(
            model_type="distilbert-base-uncased",
            lang="en",
            rescale_with_baseline=False
        )

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ):
        if not reference or not prediction:
            return {
                "metric": self.name,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

        precision, recall, f1 = self.scorer.score(
            [prediction],
            [reference]
        )

        return {
            "metric": self.name,
            "precision": round(
                precision.item(),
                4
            ),
            "recall": round(
                recall.item(),
                4
            ),
            "f1": round(
                f1.item(),
                4
            )
        }