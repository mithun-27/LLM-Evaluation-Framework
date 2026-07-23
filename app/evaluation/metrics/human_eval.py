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

    def save_csv(self, results: List[Dict], path: Path = None) -> Path:
        csv_path = path or self.output_path.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path


class SingleHumanEvaluationMetric(BaseMetric):
    """
    Collects human ratings (1-5) for a single model response.
    """

    def __init__(self):
        super().__init__(name="Human Evaluation")

    def _get_rating(self, criterion: str) -> int:
        while True:
            try:
                val = input(f"Rate {criterion} (1-5): ").strip()
                rating = int(val)
                if 1 <= rating <= 5:
                    return rating
                print("Rating must be an integer between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter an integer between 1 and 5.")

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ) -> Dict:
        prompt = kwargs.get("prompt", "")

        print("\n" + "=" * 60)
        print("SINGLE RESPONSE HUMAN EVALUATION")
        print("=" * 60)
        print(f"\nPrompt:\n{prompt}")
        print(f"\nReference Answer:\n{reference}")
        print(f"\nModel Answer:\n{prediction}")
        print("\n" + "-" * 60)

        correctness = self._get_rating("Correctness")
        relevance = self._get_rating("Relevance")
        fluency = self._get_rating("Fluency")
        helpfulness = self._get_rating("Helpfulness")

        comment = input("\nOptional comment: ").strip()

        avg_score = (correctness + relevance + fluency + helpfulness) / 4.0

        return {
            "metric": self.name,
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction,
            "correctness": correctness,
            "relevance": relevance,
            "fluency": fluency,
            "helpfulness": helpfulness,
            "comment": comment,
            "average_score": round(avg_score, 4)
        }