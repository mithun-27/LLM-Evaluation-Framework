from app.evaluation.score_calculator import ScoreCalculator


def test_score_calculator_all_metrics():
    calculator = ScoreCalculator()
    results = {
        "BLEU": {"score": 0.5},
        "ROUGE": {"rougeL": 0.6},
        "BERTScore": {"f1": 0.8},
        "Faithfulness": {"score": 0.9},
        "Hallucination": {"hallucination_score": 0.1},
        "LLM-as-a-Judge": {"overall_score": 4.0},
        "Toxicity": {"toxicity_score": 0.0},
        "Bias": {"bias_score": 0.0}
    }

    overall = calculator.calculate(results)
    # Weights:
    # bleu (0.1): 0.5 * 0.1 = 0.05
    # rougeL (0.1): 0.6 * 0.1 = 0.06
    # bertscore (0.2): 0.8 * 0.2 = 0.16
    # faithfulness (0.2): 0.9 * 0.2 = 0.18
    # hallucination (0.15): (1 - 0.1) * 0.15 = 0.135
    # judge (0.15): (4.0 / 5.0) * 0.15 = 0.12
    # toxicity (0.05): (1 - 0.0) * 0.05 = 0.05
    # bias (0.05): (1 - 0.0) * 0.05 = 0.05
    # Total sum: 0.05+0.06+0.16+0.18+0.135+0.12+0.05+0.05 = 0.805 -> 80.50%
    assert overall == 80.50


def test_score_calculator_missing_metrics():
    calculator = ScoreCalculator()
    results = {
        "BLEU": {"score": 0.5},
        "ROUGE": {"rougeL": 0.7}
    }

    overall = calculator.calculate(results)
    # Only BLEU (0.10) and ROUGE (0.10) active. Sum of weights = 0.20
    # Normalized weights: 0.5 each.
    # Score: 0.5 * 0.5 + 0.7 * 0.5 = 0.60 -> 60.00%
    assert overall == 60.00
