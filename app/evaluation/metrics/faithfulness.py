from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.evaluation.metrics.base_metric import BaseMetric


class FaithfulnessMetric(BaseMetric):
    """
    Measures whether a model answer is semantically
    supported by the provided context.
    """

    _model = None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        super().__init__(name="Faithfulness")

        if FaithfulnessMetric._model is None:
            FaithfulnessMetric._model = SentenceTransformer(
                model_name
            )

        self.model = FaithfulnessMetric._model

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
                "score": 0.0,
                "status": "Insufficient Data"
            }

        embeddings = self.model.encode(
            [context, prediction]
        )

        score = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        score = max(
            0.0,
            min(float(score), 1.0)
        )

        if score >= 0.75:
            status = "High"

        elif score >= 0.50:
            status = "Medium"

        else:
            status = "Low"

        return {
            "metric": self.name,
            "score": round(score, 4),
            "status": status
        }