from app.evaluation.metrics.base_metric import BaseMetric


class KeywordMatchMetric(BaseMetric):
    """
    Measures how many reference keywords
    are present in the model prediction.
    """

    def __init__(self):
        super().__init__(name="Keyword Match")

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ):
        reference_words = set(
            reference.lower().split()
        )

        prediction_words = set(
            prediction.lower().split()
        )

        if not reference_words:
            score = 0.0
        else:
            matched_words = (
                reference_words & prediction_words
            )

            score = len(matched_words) / len(reference_words)

        return {
            "metric": self.name,
            "score": round(score, 4)
        }