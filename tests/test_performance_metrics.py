import time

from app.evaluation.metrics.cost import CostMetric
from app.evaluation.metrics.latency import LatencyMetric
from app.utils.timer import Timer


def test_latency_metric():
    timer = Timer()

    timer.start()

    time.sleep(0.1)

    latency = timer.stop()

    metric = LatencyMetric()

    result = metric.calculate(
        reference="test",
        prediction="test",
        latency=latency
    )

    print("\nLatency Result:")
    print(result)

    assert result["latency_seconds"] >= 0.1
    assert result["latency_ms"] >= 100


def test_local_model_cost():
    metric = CostMetric()

    result = metric.calculate(
        reference="Artificial Intelligence",
        prediction=(
            "AI enables machines to perform "
            "intelligent tasks."
        ),
        prompt="What is Artificial Intelligence?"
    )

    print("\nLocal Model Cost:")
    print(result)

    assert result["estimated_cost_usd"] == 0.0
    assert result["total_tokens"] > 0


def test_custom_model_cost():
    metric = CostMetric(
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002
    )

    result = metric.calculate(
        reference="test",
        prediction="Generated model answer",
        input_tokens=1000,
        output_tokens=500
    )

    print("\nCustom Model Cost:")
    print(result)

    assert result["estimated_cost_usd"] == 0.002
    assert result["total_tokens"] == 1500