from app.evaluation.evaluator import Evaluator
from app.evaluation.metrics.bleu import BLEUMetric
from app.evaluation.metrics.rouge import ROUGEMetric
from app.evaluation.metrics.bertscore import BERTScoreMetric
from app.evaluation.metrics.latency import LatencyMetric
from app.evaluation.metrics.cost import CostMetric
from app.evaluation.metrics.faithfulness import FaithfulnessMetric
from app.evaluation.metrics.hallucination import HallucinationMetric
from app.evaluation.metrics.bias import BiasMetric
from app.evaluation.metrics.toxicity import ToxicityMetric

def test_evaluation_pipeline():
    evaluator = Evaluator(
        metrics=[
        BLEUMetric(),
        ROUGEMetric(),
        BERTScoreMetric(),
        FaithfulnessMetric(),
        HallucinationMetric(),
        ToxicityMetric(),
        BiasMetric(),
        LatencyMetric(),
        CostMetric()
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
        prediction=prediction,
        prompt="What is Artificial Intelligence?",
        context=(
        "Artificial Intelligence is the simulation "
        "of human intelligence by machines."
        ),
        latency=1.24
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