import re

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from app.evaluation.metrics.base_metric import BaseMetric


class BLEUMetric(BaseMetric):
    """
    Calculates BLEU score between a reference answer
    and a model-generated answer.
    """

    def __init__(self):
        super().__init__(name="BLEU")
        self.smoothing_function = SmoothingFunction().method1

    @staticmethod
    def tokenize(text: str):
        """
        Convert text into lowercase word tokens.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ):
        if not reference or not prediction:
            return {
                "metric": self.name,
                "score": 0.0
            }

        reference_tokens = self.tokenize(reference)
        prediction_tokens = self.tokenize(prediction)

        score = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            smoothing_function=self.smoothing_function
        )

        return {
            "metric": self.name,
            "score": round(float(score), 4)
        }