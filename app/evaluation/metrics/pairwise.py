from typing import Dict, List

from app.evaluation.metrics.base_metric import BaseMetric


class PairwiseComparisonMetric(BaseMetric):
    """
    Compares Model A and Model B using evaluation scores.
    """

    def __init__(self, tie_threshold: float = 0.01):
        super().__init__(name="Pairwise Comparison")
        self.tie_threshold = tie_threshold

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ) -> Dict:
        model_a_score = float(
            kwargs.get("model_a_score", 0.0)
        )

        model_b_score = float(
            kwargs.get("model_b_score", 0.0)
        )

        difference = model_a_score - model_b_score

        if abs(difference) <= self.tie_threshold:
            winner = "Tie"

        elif difference > 0:
            winner = "Model A"

        else:
            winner = "Model B"

        return {
            "metric": self.name,
            "model_a_score": round(model_a_score, 4),
            "model_b_score": round(model_b_score, 4),
            "score_difference": round(
                abs(difference),
                4
            ),
            "winner": winner
        }


class WinRateCalculator:
    """
    Calculates win rates from pairwise comparison results.
    """

    @staticmethod
    def calculate(results: List[Dict]) -> Dict:
        total = len(results)

        if total == 0:
            return {
                "total_comparisons": 0,
                "model_a_wins": 0,
                "model_b_wins": 0,
                "ties": 0,
                "model_a_win_rate": 0.0,
                "model_b_win_rate": 0.0,
                "tie_rate": 0.0
            }

        model_a_wins = sum(
            result["winner"] == "Model A"
            for result in results
        )

        model_b_wins = sum(
            result["winner"] == "Model B"
            for result in results
        )

        ties = sum(
            result["winner"] == "Tie"
            for result in results
        )

        return {
            "total_comparisons": total,
            "model_a_wins": model_a_wins,
            "model_b_wins": model_b_wins,
            "ties": ties,
            "model_a_win_rate": round(
                model_a_wins / total,
                4
            ),
            "model_b_win_rate": round(
                model_b_wins / total,
                4
            ),
            "tie_rate": round(
                ties / total,
                4
            )
        }