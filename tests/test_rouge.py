from app.evaluation.metrics.rouge import ROUGEMetric


def test_rouge_metric():

    metric = ROUGEMetric()

    result = metric.calculate(
        reference=(
            "Artificial Intelligence is the simulation "
            "of human intelligence by machines"
        ),
        prediction=(
            "Artificial Intelligence allows machines "
            "to simulate human intelligence"
        )
    )

    print("\nROUGE Evaluation Result")
    print("-----------------------")

    print(f"Metric  : {result['metric']}")
    print(f"ROUGE-1 : {result['rouge1']}")
    print(f"ROUGE-2 : {result['rouge2']}")
    print(f"ROUGE-L : {result['rougeL']}")

    assert result["metric"] == "ROUGE"

    assert 0.0 <= result["rouge1"] <= 1.0
    assert 0.0 <= result["rouge2"] <= 1.0
    assert 0.0 <= result["rougeL"] <= 1.0


if __name__ == "__main__":
    test_rouge_metric()