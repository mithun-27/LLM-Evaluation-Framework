import pandas as pd

from app.evaluation.metrics.bertscore import BERTScoreMetric
from app.evaluation.metrics.pairwise import (
    PairwiseComparisonMetric,
    WinRateCalculator,
)


def test_model_comparison():
    model_a = pd.read_csv("datasets/model_a.csv")
    model_b = pd.read_csv("datasets/model_b.csv")

    assert len(model_a) == len(model_b)

    bertscore = BERTScoreMetric()
    pairwise = PairwiseComparisonMetric()

    comparison_results = []

    print("\n========== MODEL COMPARISON ==========")

    for index in range(len(model_a)):
        reference = model_a.iloc[index][
            "reference_answer"
        ]

        answer_a = model_a.iloc[index]["model_answer"]
        answer_b = model_b.iloc[index]["model_answer"]

        score_a = bertscore.calculate(
            reference=reference,
            prediction=answer_a
        )["f1"]

        score_b = bertscore.calculate(
            reference=reference,
            prediction=answer_b
        )["f1"]

        comparison = pairwise.calculate(
            reference=reference,
            prediction=answer_a,
            model_a_score=score_a,
            model_b_score=score_b
        )

        comparison_results.append(comparison)

        print(f"\nQuestion {index + 1}")
        print(f"Prompt  : {model_a.iloc[index]['prompt']}")
        print(f"Model A : {score_a}")
        print(f"Model B : {score_b}")
        print(f"Winner  : {comparison['winner']}")

    win_rate = WinRateCalculator.calculate(
        comparison_results
    )

    print("\n========== WIN RATE ==========")
    print(win_rate)
    print("==============================")

    assert win_rate["total_comparisons"] == 5


if __name__ == "__main__":
    test_model_comparison()