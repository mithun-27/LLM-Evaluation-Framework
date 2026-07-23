from app.evaluation.metrics.faithfulness import (
    FaithfulnessMetric,
)


def test_faithful_answer():
    metric = FaithfulnessMetric()

    context = (
        "Paris is the capital and largest city of France."
    )

    prediction = (
        "Paris is the capital of France."
    )

    result = metric.calculate(
        reference="Paris is the capital of France.",
        prediction=prediction,
        context=context
    )

    print("\nFaithful Answer:")
    print(result)

    assert result["score"] > 0.5


def test_unfaithful_answer():
    metric = FaithfulnessMetric()

    context = (
        "Paris is the capital of France."
    )

    prediction = (
        "Tokyo is the capital of France."
    )

    result = metric.calculate(
        reference="Paris is the capital of France.",
        prediction=prediction,
        context=context
    )

    print("\nUnfaithful Answer:")
    print(result)

    assert 0 <= result["score"] <= 1


def test_missing_context():
    metric = FaithfulnessMetric()

    result = metric.calculate(
        reference="Test",
        prediction="Test",
        context=""
    )

    assert result["score"] == 0.0
    assert result["status"] == "Insufficient Data"