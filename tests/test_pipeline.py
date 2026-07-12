from app.evaluation.metrics.base_metric import BaseMetric


class TestMetric(BaseMetric):

    def __init__(self):
        super().__init__(name="Test Metric")

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


metric = TestMetric()

result = metric.calculate(
    reference="Artificial Intelligence",
    prediction="Artificial Intelligence"
)

print(result)