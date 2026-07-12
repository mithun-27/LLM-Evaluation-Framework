from unittest.mock import MagicMock, patch

from app.evaluation.metrics.llm_judge import LLMJudgeMetric


@patch("app.evaluation.metrics.llm_judge.AutoTokenizer")
@patch("app.evaluation.metrics.llm_judge.AutoModelForSeq2SeqLM")
@patch("app.evaluation.metrics.llm_judge.torch.cuda.is_available", return_value=False)
def test_llm_judge_calculate(mock_cuda, mock_model_cls, mock_tokenizer_cls):
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    mock_model_cls.from_pretrained.return_value = mock_model

    mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_tokenizer.decode.return_value = (
        '{\n  "correctness": 4,\n  "relevance": 5,\n'
        '  "completeness": 4,\n  "clarity": 5,\n'
        '  "reason": "Well written."\n}'
    )

    metric = LLMJudgeMetric(model_name="google/flan-t5-base")
    res = metric.calculate(
        reference="Paris is the capital of France.",
        prediction="Paris is France's capital.",
        prompt="What is the capital of France?"
    )

    assert res["metric"] == "LLM-as-a-Judge"
    assert res["correctness"] == 4
    assert res["relevance"] == 5
    assert res["completeness"] == 4
    assert res["clarity"] == 5
    assert res["overall_score"] == 4.5
    assert res["reason"] == "Well written."


def test_llm_judge_fallback_parsing():
    metric = LLMJudgeMetric()

    # Test JSON markdown wrapper parsing
    wrapped_json = (
        "```json\n"
        "{\n"
        '  "correctness": 5,\n'
        '  "relevance": 4,\n'
        '  "completeness": 5,\n'
        '  "clarity": 3,\n'
        '  "reason": "Parsed successfully"\n'
        "}\n"
        "```"
    )
    parsed = metric._parse_output(wrapped_json)
    assert parsed["correctness"] == 5
    assert parsed["clarity"] == 3
    assert parsed["reason"] == "Parsed successfully"

    # Test regex key extraction fallback
    non_json_text = (
        "Evaluation results: correctness is 2, relevance: 4. "
        "We note completeness = 3 and clarity is 5. "
        'reason: "Lacked detail."'
    )
    parsed_non_json = metric._parse_output(non_json_text)
    assert parsed_non_json["correctness"] == 2
    assert parsed_non_json["relevance"] == 4
    assert parsed_non_json["completeness"] == 3
    assert parsed_non_json["clarity"] == 5
    assert parsed_non_json["reason"] == "Lacked detail."
