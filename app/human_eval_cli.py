import pandas as pd

from app.evaluation.metrics.human_eval import (
    HumanEvaluationMetric,
    HumanEvaluationStore,
)
from app.evaluation.metrics.pairwise import (
    WinRateCalculator,
)


def run_human_evaluation():
    model_a = pd.read_csv(
        "datasets/model_a.csv"
    )

    model_b = pd.read_csv(
        "datasets/model_b.csv"
    )

    if len(model_a) != len(model_b):
        raise ValueError(
            "Model datasets must contain "
            "the same number of rows."
        )

    metric = HumanEvaluationMetric()

    results = []

    for index in range(len(model_a)):
        row_a = model_a.iloc[index]
        row_b = model_b.iloc[index]

        result = metric.calculate(
            reference=row_a["reference_answer"],
            prediction=row_a["model_answer"],
            prompt=row_a["prompt"],
            model_a_answer=row_a["model_answer"],
            model_b_answer=row_b["model_answer"]
        )

        results.append(result)

    store = HumanEvaluationStore()

    saved_path = store.save(results)

    win_rate = WinRateCalculator.calculate(
        results
    )

    print("\n" + "=" * 60)
    print("HUMAN EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Total Comparisons : "
        f"{win_rate['total_comparisons']}"
    )

    print(
        f"Model A Wins      : "
        f"{win_rate['model_a_wins']}"
    )

    print(
        f"Model B Wins      : "
        f"{win_rate['model_b_wins']}"
    )

    print(
        f"Ties              : "
        f"{win_rate['ties']}"
    )

    print(
        f"Model A Win Rate  : "
        f"{win_rate['model_a_win_rate']:.2%}"
    )

    print(
        f"Model B Win Rate  : "
        f"{win_rate['model_b_win_rate']:.2%}"
    )

    print(
        f"Tie Rate          : "
        f"{win_rate['tie_rate']:.2%}"
    )

    print(f"\nReport saved to: {saved_path}")

    print("=" * 60)


if __name__ == "__main__":
    run_human_evaluation()