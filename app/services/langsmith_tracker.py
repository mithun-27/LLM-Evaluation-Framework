import logging
import os
from typing import Any, Dict

logger = logging.getLogger("LLM-Evaluator")


class LangSmithTracker:
    """
    Optional LangSmith tracker for tracing.
    If environment variables are missing, disables itself gracefully.
    """

    def __init__(self):
        self.api_key = os.getenv("LANGCHAIN_API_KEY")
        self.project = os.getenv("LANGCHAIN_PROJECT", "LLM-Evaluation")
        self.tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

        if not self.api_key or not self.tracing_enabled:
            logger.info(
                "LangSmith tracing is disabled (missing LANGCHAIN_API_KEY "
                "or LANGCHAIN_TRACING_V2 is not set to true)."
            )
            self.enabled = False
        else:
            try:
                from langsmith import Client
                self.client = Client()
                self.enabled = True
                logger.info(f"LangSmith client initialized for project: {self.project}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith: {e}. Disabling LangSmith.")
                self.enabled = False

    def log_run(
        self,
        name: str,
        run_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        error: str = None
    ):
        if not self.enabled:
            return

        try:
            self.client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs,
                outputs=outputs,
                project_name=self.project,
                error=error
            )
        except Exception as e:
            logger.warning(f"Error logging run to LangSmith: {e}")
