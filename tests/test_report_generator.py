from app.services.report_generator import ReportGenerator


def test_report_generator(tmp_path):
    eval_data = {
        "summary": {
            "total_rows": 1,
            "overall_score": 85.50,
            "metrics": {
                "BLEU": {"score": 0.85}
            },
            "average_row_latency_seconds": 0.12,
            "total_evaluation_latency_seconds": 0.12
        },
        "results": [
            {
                "row_index": 0,
                "prompt": "What is AI?",
                "reference_answer": "Artificial Intelligence",
                "model_answer": "Artificial Intelligence processes",
                "context": "",
                "overall_score": 85.50,
                "metrics": {
                    "BLEU": {"score": 0.85}
                }
            }
        ]
    }

    report_gen = ReportGenerator(eval_data)

    json_file = tmp_path / "report.json"
    csv_file = tmp_path / "report.csv"

    saved_json = report_gen.to_json(str(json_file))
    saved_csv = report_gen.to_csv(str(csv_file))

    assert saved_json.exists()
    assert saved_csv.exists()

    # Ensure console print runs without error
    report_gen.print_summary()
