import json
import re
from typing import Any, Dict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

from app.evaluation.metrics.base_metric import BaseMetric


class LLMJudgeMetric(BaseMetric):
    """
    Evaluates LLM answers using a local instruction model.
    Scores correctness, relevance, completeness, and clarity (1-5).
    """

    _model = None
    _tokenizer = None
    _device = None

    def __init__(self, model_name: str = "google/flan-t5-base"):
        super().__init__(name="LLM-as-a-Judge")
        self.model_name = model_name

    def _load_model(self):
        if LLMJudgeMetric._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            LLMJudgeMetric._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if "t5" in self.model_name.lower():
                LLMJudgeMetric._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            else:
                LLMJudgeMetric._model = AutoModelForCausalLM.from_pretrained(self.model_name)

            LLMJudgeMetric._device = device
            LLMJudgeMetric._model.to(device)
            LLMJudgeMetric._model.eval()

        self.tokenizer = LLMJudgeMetric._tokenizer
        self.model = LLMJudgeMetric._model
        self.device = LLMJudgeMetric._device

    def calculate(
        self,
        reference: str,
        prediction: str,
        **kwargs
    ) -> Dict[str, Any]:
        prompt_text = kwargs.get("prompt", "")

        if not reference or not prediction or not prompt_text:
            return {
                "metric": self.name,
                "correctness": 1,
                "relevance": 1,
                "completeness": 1,
                "clarity": 1,
                "overall_score": 1.0,
                "reason": "Missing reference, prediction, or prompt."
            }

        self._load_model()

        input_text = (
            f"Evaluate the generated model answer compared to the reference answer for the given prompt.\n"
            f"Provide scores from 1 (poor) to 5 (excellent) for: correctness, relevance, completeness, clarity.\n"
            f"Prompt: {prompt_text}\n"
            f"Reference: {reference}\n"
            f"Model Answer: {prediction}\n\n"
            f"Return ONLY valid JSON format like:\n"
            f"{{\n"
            f'  "correctness": 5,\n'
            f'  "relevance": 5,\n'
            f'  "completeness": 5,\n'
            f'  "clarity": 5,\n'
            f'  "reason": "explanation"\n'
            f"}}"
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False
            )

        output_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        parsed = self._parse_output(output_text)

        overall = (
            parsed["correctness"]
            + parsed["relevance"]
            + parsed["completeness"]
            + parsed["clarity"]
        ) / 4.0

        return {
            "metric": self.name,
            "correctness": parsed["correctness"],
            "relevance": parsed["relevance"],
            "completeness": parsed["completeness"],
            "clarity": parsed["clarity"],
            "overall_score": round(overall, 4),
            "reason": parsed["reason"]
        }

    def _parse_output(self, text: str) -> Dict[str, Any]:
        # Try parsing direct JSON
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            return self._validate_and_sanitize(data)
        except Exception:
            pass

        # Try regex search for JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._validate_and_sanitize(data)
            except Exception:
                pass

        # Regex fallback key-value extraction
        scores = {
            "correctness": 3,
            "relevance": 3,
            "completeness": 3,
            "clarity": 3
        }
        reason = f"Fallback parsed from model output: {text[:100]}"

        for key in scores.keys():
            pattern = rf'"{key}"\s*:\s*(\d)|{key}\s*(?::|=|\bis\b)\s*(\d)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = next(g for g in match.groups() if g is not None)
                scores[key] = int(val)

        match_reason = re.search(
            r'"?reason"?\s*(?::|=|\bis\b)\s*"([^"]+)"',
            text,
            re.IGNORECASE
        )
        if match_reason:
            reason = match_reason.group(1)

        scores["reason"] = reason
        return self._validate_and_sanitize(scores)

    def _validate_and_sanitize(self, data: Dict) -> Dict[str, Any]:
        sanitized = {}
        for key in ["correctness", "relevance", "completeness", "clarity"]:
            val = data.get(key, 3)
            try:
                val = int(float(val))
                val = max(1, min(val, 5))
            except (ValueError, TypeError):
                val = 3
            sanitized[key] = val

        sanitized["reason"] = str(data.get("reason", "No reason provided."))
        return sanitized
