from typing import Any, Dict

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.tokenize import wordpunct_tokenize

from app.evaluation.metrics.base_metric import BaseMetric


class BLEUMetric(BaseMetric):
    """
    Calculates BLEU score between a reference answer
    and a model-generated answer.
    """

    def __init__(self):
        super().__init__(name="BLEU")
        self.smoothing_function = SmoothingFunction().method1

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ) -> Dict[str, Any]:

        if not reference or not prediction:
            return {
                "metric": self.name,
                "score": 0.0
            }

        reference_tokens = wordpunct_tokenize(reference.lower())
        prediction_tokens = wordpunct_tokenize(prediction.lower())

        score = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            smoothing_function=self.smoothing_function
        )

        return {
            "metric": self.name,
            "score": round(float(score), 4)
        }