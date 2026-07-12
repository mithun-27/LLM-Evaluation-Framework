from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SingleEvaluationRequest(BaseModel):
    prompt: str = Field(..., example="What is Artificial Intelligence?")
    prediction: str = Field(
        ...,
        example="AI enables computers to perform human tasks."
    )
    reference: str = Field(
        ...,
        example="Artificial Intelligence is the simulation of human intelligence by machines."
    )
    context: Optional[str] = Field(
        default="",
        example="AI is the simulation of human intelligence processes by machines."
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        example=["bleu", "rouge", "bias"]
    )


class SingleEvaluationResponse(BaseModel):
    overall_score: float
    metrics: Dict[str, Any]


class PairwiseComparisonRow(BaseModel):
    prompt: str
    reference_answer: str
    model_answer_a: str
    model_answer_b: str
    context: Optional[str] = ""


class ComparisonRequest(BaseModel):
    rows: List[PairwiseComparisonRow]
    metric: Optional[str] = Field(default="bertscore", example="bertscore")
    tie_threshold: Optional[float] = Field(default=0.01, example=0.01)


class ComparisonResponse(BaseModel):
    total_comparisons: int
    model_a_wins: int
    model_b_wins: int
    ties: int
    model_a_win_rate: float
    model_b_win_rate: float
    tie_rate: float
