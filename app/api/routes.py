from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ComparisonRequest,
    ComparisonResponse,
    SingleEvaluationRequest,
    SingleEvaluationResponse,
)
from app.cli import get_metrics_list
from app.evaluation.evaluator import Evaluator
from app.evaluation.score_calculator import ScoreCalculator

router = APIRouter()


@router.get("/metrics")
def list_metrics():
    """
    Get all available evaluation metrics.
    """
    return {
        "metrics": [
            {
                "key": "bleu",
                "name": "BLEU",
                "description": "N-gram overlap evaluation"
            },
            {
                "key": "rouge",
                "name": "ROUGE",
                "description": "Recall of n-grams and longest common subsequence"
            },
            {
                "key": "bertscore",
                "name": "BERTScore",
                "description": "Contextual token semantic similarity"
            },
            {
                "key": "faithfulness",
                "name": "Faithfulness",
                "description": "Semantic alignment with context"
            },
            {
                "key": "hallucination",
                "name": "Hallucination Detection",
                "description": "Contradiction detection using local NLI"
            },
            {
                "key": "toxicity",
                "name": "Toxicity Detection",
                "description": "Safety filter evaluating harmful language"
            },
            {
                "key": "bias",
                "name": "Bias Detection",
                "description": "Pattern matching for biased words"
            },
            {
                "key": "latency",
                "name": "Latency Measurement",
                "description": "Logs response latency"
            },
            {
                "key": "cost",
                "name": "Cost Estimation",
                "description": "Token usage cost approximation"
            },
            {
                "key": "judge",
                "name": "LLM-as-a-Judge",
                "description": "Instruction model correctness and clarity evaluation"
            },
            {
                "key": "custom",
                "name": "Custom Metric Example",
                "description": "Keyword match score"
            }
        ]
    }


@router.post("/evaluate", response_model=SingleEvaluationResponse)
def evaluate_single(payload: SingleEvaluationRequest):
    """
    Evaluate a single prediction against prompt and reference.
    """
    metric_keys = ",".join(payload.metrics) if payload.metrics else None
    try:
        metrics = get_metrics_list(metric_keys)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    evaluator = Evaluator(metrics)

    results = evaluator.evaluate(
        reference=payload.reference,
        prediction=payload.prediction,
        prompt=payload.prompt,
        context=payload.context
    )

    calculator = ScoreCalculator()
    overall = calculator.calculate(results)

    return {
        "overall_score": overall,
        "metrics": results
    }


@router.post("/compare", response_model=ComparisonResponse)
def compare_models(payload: ComparisonRequest):
    """
    Compare Model A and Model B outputs across multiple prompt samples.
    """
    from app.evaluation.metrics.pairwise import (
        PairwiseComparisonMetric,
        WinRateCalculator,
    )

    if not payload.rows:
        raise HTTPException(
            status_code=400,
            detail="Comparison row list cannot be empty."
        )

    metrics_list = get_metrics_list(payload.metric)
    if not metrics_list:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown comparison metric '{payload.metric}'"
        )

    scorer = metrics_list[0]
    pairwise = PairwiseComparisonMetric(tie_threshold=payload.tie_threshold)

    comparison_results = []
    for row in payload.rows:
        res_a = scorer.calculate(
            reference=row.reference_answer,
            prediction=row.model_answer_a,
            prompt=row.prompt,
            context=row.context
        )
        res_b = scorer.calculate(
            reference=row.reference_answer,
            prediction=row.model_answer_b,
            prompt=row.prompt,
            context=row.context
        )

        def extract_score(res):
            if "f1" in res:
                return res["f1"]
            if "score" in res:
                return res["score"]
            if "rougeL" in res:
                return res["rougeL"]
            return 0.0

        score_a = extract_score(res_a)
        score_b = extract_score(res_b)

        comparison = pairwise.calculate(
            reference=row.reference_answer,
            prediction=row.model_answer_a,
            model_a_score=score_a,
            model_b_score=score_b
        )
        comparison_results.append(comparison)

    win_rate = WinRateCalculator.calculate(comparison_results)
    return win_rate
