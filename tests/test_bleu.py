from app.evaluation.metrics.bleu import BLEUMetric


def test_bleu_metric():
    metric = BLEUMetric()

    reference = (
        "Artificial Intelligence enables machines to learn"
    )

    prediction = (
        "Artificial Intelligence enables machines to learn"
    )

    result = metric.calculate(
        reference=reference,
        prediction=prediction
    )

    print("\nBLEU Result:")
    print(result)

    assert result["metric"] == "BLEU"
    assert 0 <= result["score"] <= 1
    assert result["score"] == 1.0


def test_bleu_different_sentence():
    metric = BLEUMetric()

    reference = "Paris is the capital of France"

    prediction = "France has a city called Paris"

    result = metric.calculate(
        reference=reference,
        prediction=prediction
    )

    print("\nDifferent Sentence BLEU Result:")
    print(result)

    assert 0 <= result["score"] <= 1


def test_bleu_empty_prediction():
    metric = BLEUMetric()

    result = metric.calculate(
        reference="Artificial Intelligence",
        prediction=""
    )

    assert result["score"] == 0.0