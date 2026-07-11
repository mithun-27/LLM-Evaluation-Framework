from app.evaluation.metrics.base_metric import BaseMetric


class LatencyMetric(BaseMetric):
    """
    Measures model response latency.
    """

    def __init__(self):
        super().__init__(name="Latency")

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ):
        latency = kwargs.get("latency", 0.0)

        latency = max(float(latency), 0.0)

        return {
            "metric": self.name,
            "latency_seconds": round(latency, 4),
            "latency_ms": round(latency * 1000, 2)
        }