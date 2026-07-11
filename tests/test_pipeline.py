from app.evaluation.evaluator import Evaluator
from app.evaluation.metrics.bleu import BLEUMetric
from app.evaluation.metrics.rouge import ROUGEMetric
from app.evaluation.metrics.bertscore import BERTScoreMetric


def test_evaluation_pipeline():
    evaluator = Evaluator(
        metrics=[
            BLEUMetric(),
            ROUGEMetric(),
            BERTScoreMetric()
        ]
    )

    reference = (
        "Artificial Intelligence is the simulation "
        "of human intelligence by machines"
    )

    prediction = (
        "AI enables computers to imitate "
        "human intelligence"
    )

    results = evaluator.evaluate(
        reference=reference,
        prediction=prediction
    )

    print("\n========== LLM Evaluation ==========")

    for metric_name, result in results.items():
        print(f"\n{metric_name}")
        print(result)

    print("\n====================================")

    assert "BLEU" in results
    assert "ROUGE" in results
    assert "BERTScore" in results


if __name__ == "__main__":
    test_evaluation_pipeline()