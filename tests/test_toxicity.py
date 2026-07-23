from app.evaluation.metrics.toxicity import ToxicityMetric


def test_safe_response():
    metric = ToxicityMetric()

    result = metric.calculate(
        reference="Be helpful.",
        prediction=(
            "I would be happy to help you "
            "understand this topic."
        )
    )

    print("\nSafe Response:")
    print(result)

    assert result["status"] == "Safe"
    assert 0 <= result["toxicity_score"] <= 1


def test_toxic_response():
    metric = ToxicityMetric()

    result = metric.calculate(
        reference="Respond politely.",
        prediction=(
            "You are an idiot and I hate you."
        )
    )

    print("\nToxic Response:")
    print(result)

    assert 0 <= result["toxicity_score"] <= 1


def test_empty_response():
    metric = ToxicityMetric()

    result = metric.calculate(
        reference="Test",
        prediction=""
    )

    assert result["toxicity_score"] == 0.0
    assert result["status"] == "Insufficient Data"