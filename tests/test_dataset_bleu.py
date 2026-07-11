from app.evaluation.metrics.bleu import BLEUMetric
from app.services.dataset_loader import DatasetLoader


def test_dataset_bleu():
    loader = DatasetLoader(
        "datasets/sample_dataset.csv"
    )

    dataframe = loader.load()

    metric = BLEUMetric()

    print("\n========== BLEU Evaluation ==========")

    for index, row in dataframe.iterrows():
        result = metric.calculate(
            reference=row["reference_answer"],
            prediction=row["model_answer"]
        )

        print(f"\nQuestion {index + 1}")
        print(f"Prompt : {row['prompt']}")
        print(f"BLEU   : {result['score']}")

    print("\n=====================================")


if __name__ == "__main__":
    test_dataset_bleu()