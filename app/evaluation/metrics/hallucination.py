import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.evaluation.metrics.base_metric import BaseMetric


class HallucinationMetric(BaseMetric):
    """
    Detects whether a model response is supported,
    contradicted, or unsupported by the context.
    """

    _model = None
    _tokenizer = None

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small"
    ):
        super().__init__(name="Hallucination")

        if HallucinationMetric._tokenizer is None:
            HallucinationMetric._tokenizer = (
                AutoTokenizer.from_pretrained(model_name)
            )

        if HallucinationMetric._model is None:
            HallucinationMetric._model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
            )

            HallucinationMetric._model.eval()

        self.tokenizer = HallucinationMetric._tokenizer
        self.model = HallucinationMetric._model

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ):
        context = kwargs.get("context", "")

        if not context or not prediction:
            return {
                "metric": self.name,
                "hallucination_score": 0.0,
                "label": "Insufficient Data"
            }

        inputs = self.tokenizer(
            context,
            prediction,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        contradiction = probabilities[0].item()
        entailment = probabilities[1].item()
        neutral = probabilities[2].item()

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

        labels = [
            "Contradiction",
            "Entailment",
            "Neutral"
        ]

        predicted_label = labels[predicted_index]

        hallucination_score = (
            contradiction + neutral
        )

        if predicted_label == "Entailment":
            status = "Supported"

        elif predicted_label == "Contradiction":
            status = "Hallucinated"

        else:
            status = "Potential Hallucination"

        return {
            "metric": self.name,
            "hallucination_score": round(
                hallucination_score,
                4
            ),
            "entailment_score": round(
                entailment,
                4
            ),
            "contradiction_score": round(
                contradiction,
                4
            ),
            "neutral_score": round(
                neutral,
                4
            ),
            "label": predicted_label,
            "status": status
        }