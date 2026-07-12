import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class ReportGenerator:
    """
    Generates JSON, CSV, and formatted Console reports from evaluation results.
    """

    def __init__(self, evaluation_data: Dict[str, Any]):
        self.summary = evaluation_data.get("summary", {})
        self.results = evaluation_data.get("results", [])
        self.console = Console()

    def to_json(self, output_path: str) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "summary": self.summary,
            "results": self.results
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        return path

    def to_csv(self, output_path: str) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for res in self.results:
            row_dict = {
                "row_index": res.get("row_index"),
                "prompt": res.get("prompt"),
                "reference_answer": res.get("reference_answer"),
                "model_answer": res.get("model_answer"),
                "context": res.get("context"),
                "overall_score": res.get("overall_score"),
            }
            for metric_name, m_data in res.get("metrics", {}).items():
                if "error" in m_data:
                    row_dict[f"{metric_name}_error"] = m_data["error"]
                else:
                    for key, val in m_data.items():
                        if key != "metric":
                            row_dict[f"{metric_name}_{key}"] = val
            rows.append(row_dict)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8")
        return path

    def print_summary(self):
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold green]LLM EVALUATION SUMMARY REPORT[/bold green]",
            subtitle=f"Evaluated {self.summary.get('total_rows', 0)} Rows"
        ))

        table = Table(
            title="Aggregate Metrics",
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Sub-Metric / Property", style="magenta")
        table.add_column("Value", justify="right", style="green")

        for metric_name, sub_metrics in self.summary.get("metrics", {}).items():
            for key, val in sub_metrics.items():
                table.add_row(metric_name, key, f"{val:.4f}")

        self.console.print(table)

        perf_table = Table(
            title="Performance & Cost",
            show_header=True,
            header_style="bold yellow"
        )
        perf_table.add_column("Parameter", style="cyan")
        perf_table.add_column("Value", justify="right", style="green")

        perf_table.add_row(
            "Average Row Latency",
            f"{self.summary.get('average_row_latency_seconds', 0.0):.4f} seconds"
        )
        perf_table.add_row(
            "Total Evaluation Latency",
            f"{self.summary.get('total_evaluation_latency_seconds', 0.0):.4f} seconds"
        )

        cost_sum = 0.0
        if "Cost" in self.summary.get("metrics", {}):
            cost_sum = self.summary["metrics"]["Cost"].get("estimated_cost_usd", 0.0)
        perf_table.add_row("Total Estimated Cost (USD)", f"${cost_sum:.6f}")

        self.console.print(perf_table)

        overall_score = self.summary.get("overall_score", 0.0)
        self.console.print(
            f"\n[bold green]OVERALL SCORE: {overall_score:.2f} / 100[/bold green]"
        )
        self.console.print("=" * 60 + "\n")
