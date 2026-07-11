import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from app.evaluation.metrics.base_metric import BaseMetric


class HumanEvaluationMetric(BaseMetric):
    """
    Collects human preference between two model responses.
    """

    def __init__(self):
        super().__init__(name="Human Evaluation")

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ) -> Dict:
        prompt = kwargs.get("prompt", "")
        model_a_answer = kwargs.get(
            "model_a_answer",
            prediction
        )
        model_b_answer = kwargs.get(
            "model_b_answer",
            ""
        )

        print("\n" + "=" * 60)
        print("HUMAN EVALUATION")
        print("=" * 60)

        print(f"\nPrompt:\n{prompt}")

        print(f"\nReference Answer:\n{reference}")

        print(f"\nModel A:\n{model_a_answer}")

        print(f"\nModel B:\n{model_b_answer}")

        print("\nChoose the better response:")
        print("[1] Model A")
        print("[2] Model B")
        print("[3] Tie")

        while True:
            choice = input("\nYour choice: ").strip()

            if choice == "1":
                winner = "Model A"
                break

            if choice == "2":
                winner = "Model B"
                break

            if choice == "3":
                winner = "Tie"
                break

            print(
                "Invalid choice. Please enter 1, 2, or 3."
            )

        comment = input(
            "Optional comment: "
        ).strip()

        return {
            "metric": self.name,
            "prompt": prompt,
            "reference": reference,
            "model_a_answer": model_a_answer,
            "model_b_answer": model_b_answer,
            "winner": winner,
            "comment": comment
        }


class HumanEvaluationStore:
    """
    Saves human evaluation results to JSON.
    """

    def __init__(
        self,
        output_path: str = (
            "reports/json/human_evaluations.json"
        )
    ):
        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(self, results: List[Dict]) -> Path:
        report = {
            "created_at": datetime.now().isoformat(),
            "total_evaluations": len(results),
            "evaluations": results
        }

        with self.output_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False
            )

        return self.output_path