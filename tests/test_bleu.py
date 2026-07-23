from app.evaluation.metrics.bleu import BLEUMetric


def test_bleu_metric():

    metric = BLEUMetric()

    result = metric.calculate(
        reference="Artificial Intelligence is the simulation of human intelligence by machines",
        prediction="Artificial Intelligence is the simulation of human intelligence by machines"
    )

    print("\nBLEU Evaluation Result")
    print("----------------------")
    print(f"Metric : {result['metric']}")
    print(f"Score  : {result['score']}")

    assert result["metric"] == "BLEU"
    assert result["score"] == 1.0


if __name__ == "__main__":
    test_bleu_metric()