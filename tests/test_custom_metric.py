from app.evaluation.metrics.custom_metric import KeywordMatchMetric


def test_keyword_match_metric():
    metric = KeywordMatchMetric()

    # Exact match of all reference words (case insensitive)
    res = metric.calculate(
        reference="Artificial Intelligence",
        prediction="Artificial Intelligence is the simulation of human intelligence."
    )

    assert res["metric"] == "Keyword Match"
    assert res["score"] == 1.0

    # Partial match
    res_partial = metric.calculate(
        reference="Paris France Capital",
        prediction="Paris is a city in Europe."
    )

    # Reference words: {'paris', 'france', 'capital'} -> 3 words
    # Prediction words: {'paris', 'is', 'a', 'city', 'in', 'europe'}
    # Matched: {'paris'} -> 1 word. Score = 1/3 = 0.3333
    assert res_partial["score"] == 0.3333
