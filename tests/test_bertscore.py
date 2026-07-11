from app.evaluation.metrics.bertscore import BERTScoreMetric


def test_bertscore_identical():
    metric = BERTScoreMetric()

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

    print("\nBERTScore Identical Result:")
    print(result)

    assert result["metric"] == "BERTScore"
    assert result["f1"] > 0.99


def test_bertscore_semantic_similarity():
    metric = BERTScoreMetric()

    reference = (
        "Artificial Intelligence allows machines "
        "to perform intelligent tasks"
    )

    prediction = (
        "AI enables computers to carry out tasks "
        "that require intelligence"
    )

    result = metric.calculate(
        reference=reference,
        prediction=prediction
    )

    print("\nBERTScore Semantic Result:")
    print(result)

    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


def test_bertscore_empty_prediction():
    metric = BERTScoreMetric()

    result = metric.calculate(
        reference="Artificial Intelligence",
        prediction=""
    )

    assert result["f1"] == 0.0