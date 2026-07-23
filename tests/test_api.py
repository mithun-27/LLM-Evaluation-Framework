from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}


def test_list_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert any(m["key"] == "bleu" for m in data["metrics"])


def test_evaluate_single_endpoint():
    # Call using a fast metric to avoid slow downloads during pytest collection
    payload = {
        "prompt": "What is AI?",
        "reference": "Artificial Intelligence is the simulation of human intelligence.",
        "prediction": "Artificial Intelligence is the simulation of human intelligence.",
        "metrics": ["bleu"]
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data
    assert "BLEU" in data["metrics"]
    assert data["metrics"]["BLEU"]["score"] == 1.0


def test_compare_endpoint():
    payload = {
        "rows": [
            {
                "prompt": "Capital of France?",
                "reference_answer": "Paris is the capital of France.",
                "model_answer_a": "Paris is the capital of France.",
                "model_answer_b": "France has Paris as its capital city."
            }
        ],
        "metric": "bleu",
        "tie_threshold": 0.01
    }
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_comparisons"] == 1
    assert "model_a_wins" in data
