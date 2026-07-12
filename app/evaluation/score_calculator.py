from typing import Dict, Any

DEFAULT_WEIGHTS = {
    "bleu": 0.10,
    "rougeL": 0.10,
    "bertscore_f1": 0.20,
    "faithfulness": 0.20,
    "hallucination_safety": 0.15,
    "judge_score": 0.15,
    "toxicity_safety": 0.05,
    "bias_safety": 0.05
}


class ScoreCalculator:
    """
    Normalizes individual metrics and computes an overall weighted quality score.
    Handles missing metrics dynamically by re-normalizing weights.
    Supports custom/unknown metrics by automatically assigning a default weight.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS

    def calculate(self, results: Dict[str, Any]) -> float:
        extracted = {}

        # 1. BLEU
        if "BLEU" in results and "score" in results["BLEU"]:
            extracted["bleu"] = float(results["BLEU"]["score"])

        # 2. ROUGE
        if "ROUGE" in results and "rougeL" in results["ROUGE"]:
            extracted["rougeL"] = float(results["ROUGE"]["rougeL"])

        # 3. BERTScore
        if "BERTScore" in results and "f1" in results["BERTScore"]:
            extracted["bertscore_f1"] = float(results["BERTScore"]["f1"])

        # 4. Faithfulness
        if "Faithfulness" in results and "score" in results["Faithfulness"]:
            extracted["faithfulness"] = float(results["Faithfulness"]["score"])

        # 5. Hallucination (Safety = 1 - score)
        if "Hallucination" in results and "hallucination_score" in results["Hallucination"]:
            score = float(results["Hallucination"]["hallucination_score"])
            extracted["hallucination_safety"] = max(0.0, 1.0 - score)

        # 6. LLM Judge (Normalized 1-5 to 0-1 range via score / 5.0)
        if "LLM-as-a-Judge" in results and "overall_score" in results["LLM-as-a-Judge"]:
            score = float(results["LLM-as-a-Judge"]["overall_score"])
            extracted["judge_score"] = score / 5.0

        # 7. Toxicity (Safety = 1 - score)
        if "Toxicity" in results and "toxicity_score" in results["Toxicity"]:
            score = float(results["Toxicity"]["toxicity_score"])
            extracted["toxicity_safety"] = max(0.0, 1.0 - score)

        # 8. Bias (Safety = 1 - score)
        if "Bias" in results and "bias_score" in results["Bias"]:
            score = float(results["Bias"]["bias_score"])
            extracted["bias_safety"] = max(0.0, 1.0 - score)

        # 9. Custom/Unknown Metrics
        standard_keys = [
            "BLEU", "ROUGE", "BERTScore", "Faithfulness",
            "Hallucination", "LLM-as-a-Judge", "Toxicity", "Bias"
        ]
        for metric_name, m_data in results.items():
            if metric_name in standard_keys:
                continue
            if isinstance(m_data, dict) and "score" in m_data:
                try:
                    extracted[metric_name] = float(m_data["score"])
                except (ValueError, TypeError):
                    pass

        if not extracted:
            return 0.0

        # Filter weights to only include present metrics
        active_weights = {}
        for k in extracted.keys():
            if k in self.weights:
                active_weights[k] = self.weights[k]
            else:
                # Assign a default weight of 0.10 for custom metrics
                active_weights[k] = 0.10

        weight_sum = sum(active_weights.values())
        if weight_sum == 0:
            return 0.0

        # Normalize active weights and calculate weighted sum
        weighted_score = 0.0
        for key, val in extracted.items():
            if key in active_weights:
                norm_weight = active_weights[key] / weight_sum
                weighted_score += val * norm_weight

        # Return score out of 100
        return round(weighted_score * 100, 2)
