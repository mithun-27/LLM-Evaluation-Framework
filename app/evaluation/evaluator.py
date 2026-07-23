from typing import List, Dict, Any
import pandas as pd

from app.evaluation.metrics.base_metric import BaseMetric
from app.evaluation.score_calculator import ScoreCalculator
from app.utils.timer import Timer


class Evaluator:
    """
    Main LLM evaluation engine. Supports single prediction and dataset evaluations.
    """

    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics

    def evaluate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ) -> Dict[str, Any]:
        results = {}

        for metric in self.metrics:
            try:
                metric_result = metric.calculate(
                    reference=reference,
                    prediction=prediction,
                    **kwargs
                )

                results[metric.name] = metric_result

            except Exception as error:
                results[metric.name] = {
                    "metric": metric.name,
                    "error": str(error)
                }

        return results

    def evaluate_dataset(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        calculator = ScoreCalculator()
        timer = Timer()
        timer.start()

        row_results = []

        for idx, row in df.iterrows():
            prompt = row.get("prompt", "")
            reference = row.get("reference_answer", "")
            prediction = row.get("model_answer", "")
            context = row.get("context", "")

            row_kwargs = {**kwargs, "prompt": prompt, "context": context}
            for col in df.columns:
                if col not in ["prompt", "reference_answer", "model_answer", "context"]:
                    row_kwargs[col] = row[col]

            row_metrics = self.evaluate(
                reference=reference,
                prediction=prediction,
                **row_kwargs
            )

            overall_score = calculator.calculate(row_metrics)

            record = {
                "row_index": idx,
                "prompt": prompt,
                "reference_answer": reference,
                "model_answer": prediction,
                "context": context,
                "metrics": row_metrics,
                "overall_score": overall_score
            }
            row_results.append(record)

        total_latency = timer.stop()
        avg_latency = total_latency / len(df) if len(df) > 0 else 0.0

        summary = self._calculate_summary(row_results, avg_latency, total_latency)

        return {
            "summary": summary,
            "results": row_results
        }

    def _calculate_summary(
        self,
        row_results: List[Dict],
        avg_latency: float,
        total_latency: float
    ) -> Dict[str, Any]:
        total = len(row_results)
        if total == 0:
            return {"overall_score": 0.0, "total_rows": 0}

        avg_overall = sum(r["overall_score"] for r in row_results) / total

        sums = {}
        counts = {}

        for r in row_results:
            metrics = r["metrics"]
            for metric_name, m_data in metrics.items():
                if "error" in m_data:
                    continue
                for key, val in m_data.items():
                    if key == "metric" or isinstance(val, (str, list)):
                        continue
                    try:
                        f_val = float(val)
                        sums[(metric_name, key)] = sums.get((metric_name, key), 0.0) + f_val
                        counts[(metric_name, key)] = counts.get((metric_name, key), 0) + 1
                    except (ValueError, TypeError):
                        pass

        metric_averages = {}
        for (metric_name, key), s_val in sums.items():
            cnt = counts[(metric_name, key)]
            avg = round(s_val / cnt, 4) if cnt > 0 else 0.0
            if metric_name not in metric_averages:
                metric_averages[metric_name] = {}
            metric_averages[metric_name][key] = avg

        return {
            "total_rows": total,
            "overall_score": round(avg_overall, 2),
            "metrics": metric_averages,
            "average_row_latency_seconds": round(avg_latency, 4),
            "total_evaluation_latency_seconds": round(total_latency, 4)
        }