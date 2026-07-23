import logging
import os
from typing import Any, Dict, List

import mlflow

logger = logging.getLogger("LLM-Evaluator")


class MLflowTracker:
    """
    Tracks evaluation experiments locally using MLflow.
    Logs run parameters, metric summaries, and report artifacts.
    """

    def __init__(self, experiment_name: str = "LLM-Evaluation"):
        self.experiment_name = experiment_name
        try:
            mlflow.set_experiment(experiment_name)
            self.enabled = True
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}. Tracking will be disabled.")
            self.enabled = False

    def log_evaluation(
        self,
        dataset_name: str,
        summary: Dict[str, Any],
        enabled_metrics: List[str],
        report_path: str = None
    ):
        if not self.enabled:
            logger.info("MLflow tracking is disabled. Skipping log.")
            return

        try:
            with mlflow.start_run():
                # Log params
                mlflow.log_param("dataset_name", dataset_name)
                mlflow.log_param("enabled_metrics", ",".join(enabled_metrics))
                mlflow.log_param("total_rows", summary.get("total_rows", 0))

                # Log overall metrics
                mlflow.log_metric("overall_score", summary.get("overall_score", 0.0))
                mlflow.log_metric(
                    "average_row_latency",
                    summary.get("average_row_latency_seconds", 0.0)
                )
                mlflow.log_metric(
                    "total_latency",
                    summary.get("total_evaluation_latency_seconds", 0.0)
                )

                # Log sub-metric averages dynamically
                for metric_name, sub_metrics in summary.get("metrics", {}).items():
                    for sub_key, val in sub_metrics.items():
                        # Standardize metric naming for MLflow (e.g. BLEU_score)
                        name = f"{metric_name}_{sub_key}".replace(" ", "_")
                        mlflow.log_metric(name, float(val))

                # Log report file as an artifact
                if report_path and os.path.exists(report_path):
                    mlflow.log_artifact(report_path)

                logger.info("Successfully logged evaluation results to MLflow.")

        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
