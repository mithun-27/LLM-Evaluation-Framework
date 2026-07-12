from app.evaluation.metrics.pairwise import (
    PairwiseComparisonMetric,
    WinRateCalculator,
)


def test_model_a_wins():
    metric = PairwiseComparisonMetric()

    result = metric.calculate(
        reference="test",
        prediction="test",
        model_a_score=0.90,
        model_b_score=0.70
    )

    print("\nModel A Comparison:")
    print(result)

    assert result["winner"] == "Model A"


def test_model_b_wins():
    metric = PairwiseComparisonMetric()

    result = metric.calculate(
        reference="test",
        prediction="test",
        model_a_score=0.60,
        model_b_score=0.85
    )

    print("\nModel B Comparison:")
    print(result)

    assert result["winner"] == "Model B"


def test_pairwise_tie():
    metric = PairwiseComparisonMetric()

    result = metric.calculate(
        reference="test",
        prediction="test",
        model_a_score=0.80,
        model_b_score=0.805
    )

    print("\nTie Comparison:")
    print(result)

    assert result["winner"] == "Tie"


def test_win_rate():
    results = [
        {"winner": "Model A"},
        {"winner": "Model A"},
        {"winner": "Model B"},
        {"winner": "Tie"},
        {"winner": "Model A"},
    ]

    result = WinRateCalculator.calculate(results)

    print("\nWin Rate:")
    print(result)

    assert result["total_comparisons"] == 5
    assert result["model_a_wins"] == 3
    assert result["model_b_wins"] == 1
    assert result["ties"] == 1
    assert result["model_a_win_rate"] == 0.6
    assert result["model_b_win_rate"] == 0.2
    assert result["tie_rate"] == 0.2


def test_empty_win_rate():
    result = WinRateCalculator.calculate([])

    assert result["total_comparisons"] == 0
    assert result["model_a_win_rate"] == 0.0