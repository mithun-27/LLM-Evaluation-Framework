from typing import Any, Dict, List

from app.services.langsmith_tracker import LangSmithTracker
from app.services.mlflow_tracker import MLflowTracker


class ExperimentTracker:
    """
    Unified entry point for both MLflow and LangSmith evaluation logging.
    """

    def __init__(self, experiment_name: str = "LLM-Evaluation"):
        self.mlflow_tracker = MLflowTracker(experiment_name)
        self.langsmith_tracker = LangSmithTracker()

    def log_evaluation(
        self,
        dataset_name: str,
        summary: Dict[str, Any],
        enabled_metrics: List[str],
        report_path: str = None
    ):
        # Log to MLflow
        self.mlflow_tracker.log_evaluation(
            dataset_name=dataset_name,
            summary=summary,
            enabled_metrics=enabled_metrics,
            report_path=report_path
        )

        # Log trace metadata summary to LangSmith
        self.langsmith_tracker.log_run(
            name=f"Dataset Evaluation - {dataset_name}",
            run_type="chain",
            inputs={
                "dataset_name": dataset_name,
                "enabled_metrics": enabled_metrics
            },
            outputs={"summary": summary}
        )
