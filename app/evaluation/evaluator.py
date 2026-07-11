from typing import List

from app.evaluation.metrics.base_metric import BaseMetric


class Evaluator:
    """
    Main LLM evaluation engine.
    """

    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics

    def evaluate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ):
        results = {}

        for metric in self.metrics:
            try:
                metric_result = metric.calculate(
                    reference=reference,
                    prediction=prediction,
                    **kwargs
                )

                results[metric.name] = metric_result

            except Exception as error:
                results[metric.name] = {
                    "metric": metric.name,
                    "error": str(error)
                }

        return results