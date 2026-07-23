from app.evaluation.metrics.bias import BiasMetric


def test_neutral_response():
    metric = BiasMetric()

    result = metric.calculate(
        reference="People have different abilities.",
        prediction=(
            "People should be evaluated based "
            "on their individual skills."
        )
    )

    print("\nNeutral Response:")
    print(result)

    assert result["status"] == "No Bias Detected"
    assert result["bias_score"] == 0.0


def test_potential_bias():
    metric = BiasMetric()

    result = metric.calculate(
        reference="Avoid stereotypes.",
        prediction=(
            "All women are naturally bad "
            "at technical work."
        )
    )

    print("\nPotential Bias:")
    print(result)

    assert result["status"] == "Potential Bias"
    assert result["bias_score"] > 0


def test_empty_response():
    metric = BiasMetric()

    result = metric.calculate(
        reference="Test",
        prediction=""
    )

    assert result["status"] == "Insufficient Data"