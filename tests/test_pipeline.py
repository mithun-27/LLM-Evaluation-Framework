from app.evaluation.metrics.custom_metric import KeywordMatchMetric


def test_keyword_metric():
    metric = KeywordMatchMetric()

    reference = (
        "Artificial Intelligence enables machines to learn"
    )

    prediction = (
        "Artificial Intelligence helps machines learn"
    )

    result = metric.calculate(
        reference=reference,
        prediction=prediction
    )

    print("\nMetric Result")
    print(result)

    assert "metric" in result
    assert "score" in result
    assert 0 <= result["score"] <= 1


if __name__ == "__main__":
    test_keyword_metric()