from app.evaluation.metrics.hallucination import (
    HallucinationMetric,
)


def test_supported_answer():
    metric = HallucinationMetric()

    result = metric.calculate(
        reference="Paris is the capital of France.",
        prediction="Paris is the capital of France.",
        context=(
            "Paris is the capital and largest city "
            "of France."
        )
    )

    print("\nSupported Answer:")
    print(result)

    assert result["label"] == "Entailment"
    assert result["status"] == "Supported"


def test_contradicted_answer():
    metric = HallucinationMetric()

    result = metric.calculate(
        reference="Paris is the capital of France.",
        prediction="Tokyo is the capital of France.",
        context="Paris is the capital of France."
    )

    print("\nContradicted Answer:")
    print(result)

    assert result["label"] == "Contradiction"
    assert result["status"] == "Hallucinated"


def test_missing_context():
    metric = HallucinationMetric()

    result = metric.calculate(
        reference="Test",
        prediction="Generated answer",
        context=""
    )

    assert result["label"] == "Insufficient Data"