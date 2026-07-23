from app.evaluation.metrics.base_metric import BaseMetric


class CostMetric(BaseMetric):
    """
    Estimates model inference cost using token usage.

    Local models can use zero pricing.
    """

    def __init__(
        self,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0
    ):
        super().__init__(name="Cost")

        self.input_cost_per_1k = input_cost_per_1k
        self.output_cost_per_1k = output_cost_per_1k

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Simple token approximation.

        Approximately 1 token per 4 characters.
        """

        if not text:
            return 0

        return max(1, round(len(text) / 4))

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ):
        prompt = kwargs.get("prompt", "")

        input_tokens = kwargs.get(
            "input_tokens",
            self.estimate_tokens(prompt)
        )

        output_tokens = kwargs.get(
            "output_tokens",
            self.estimate_tokens(prediction)
        )

        input_cost = (
            input_tokens / 1000
        ) * self.input_cost_per_1k

        output_cost = (
            output_tokens / 1000
        ) * self.output_cost_per_1k

        total_cost = input_cost + output_cost

        return {
            "metric": self.name,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(
                input_tokens + output_tokens
            ),
            "estimated_cost_usd": round(total_cost, 8)
        }