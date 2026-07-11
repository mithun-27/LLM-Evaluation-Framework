from app.evaluation.metrics.rouge import ROUGEMetric


def test_rouge_identical_sentence():
    metric = ROUGEMetric()

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

    print("\nROUGE Identical Result:")
    print(result)

    assert result["metric"] == "ROUGE"
    assert result["rouge1"] == 1.0
    assert result["rouge2"] == 1.0
    assert result["rougeL"] == 1.0


def test_rouge_different_sentence():
    metric = ROUGEMetric()

    reference = "Paris is the capital of France"

    prediction = "France has a city called Paris"

    result = metric.calculate(
        reference=reference,
        prediction=prediction
    )

    print("\nROUGE Different Result:")
    print(result)

    assert 0 <= result["rouge1"] <= 1
    assert 0 <= result["rouge2"] <= 1
    assert 0 <= result["rougeL"] <= 1


def test_rouge_empty_prediction():
    metric = ROUGEMetric()

    result = metric.calculate(
        reference="Artificial Intelligence",
        prediction=""
    )

    assert result["rouge1"] == 0.0
    assert result["rouge2"] == 0.0
    assert result["rougeL"] == 0.0