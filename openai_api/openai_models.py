# from __future__ import annotations
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

# -------- API families & model capability registry --------


class ApiFamily(str, Enum):
    COMPLETIONS = "completions"  # Legacy: https://platform.openai.com/docs/api-reference/completions
    RESPONSES = "responses"  # Newer API

class ModelSpec(BaseModel):
    name: str
    api_family: ApiFamily
    # Capability flags
    supports_reasoning: bool = False  # GPT-5 models
    uses_max_output_tokens: bool = False  # Responses API
    # Optional model-specific defaults
    default_reasoning_effort: Optional[str] = (
        None  # "minimal" | "low" | "medium" | "high"
    )

    @staticmethod
    def detect_model(model_name: str) -> "ModelSpec":
        mn = model_name.lower()
        if mn.startswith("gpt-5"):
            # Responses API + reasoning controls
            return ModelSpec(
                name=model_name,
                api_family=ApiFamily.RESPONSES,
                supports_reasoning=True,
                uses_max_output_tokens=True,
                default_reasoning_effort="minimal",  # good default for speed
            )
        # Everything else here falls back to Chat Completions
        return ModelSpec(
            name=model_name,
            api_family=ApiFamily.COMPLETIONS,
            supports_reasoning=False,
            uses_max_output_tokens=False,
        )

    # -------- Rendering helpers to minimize compatibility bugs --------

    def build_payload(
        self,
        *,
        system_message: str,
        task_text: str,
        response_format: Optional[BaseModel] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[
            str
        ] = None,  # "minimal" | "low" | "medium" | "high" (GPT-5)
        verbosity: Optional[str] = None,  # "low" | "medium" | "high" (GPT-5)
        extra: Optional[Dict[str, Any]] = None,  # escape hatch for rare params
    ) -> Dict[str, Any]:
        """
        Returns kwargs ready for:
          - client.chat.completions.create(**kwargs)  OR
          - client.responses.create(**kwargs)
        """
        payload: Dict[str, Any] = {"model": self.name}


        if self.api_family == ApiFamily.COMPLETIONS: #Legacy인 경우
            payload["input"] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": task_text},
            ]
            if response_format:
                payload["text"] = {
                    "format": {"type": "json_object"}
                }  # According to https://platform.openai.com/docs/guides/structured-outputs#json-mode
                # payload["response_format"] = response_format
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
        else:
            # Responses API
            payload["instructions"] = system_message
            payload["input"] = task_text
            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens
            # GPT-5 reasoning controls
            if self.supports_reasoning:
                eff = reasoning_effort or self.default_reasoning_effort
                if eff:
                    payload["reasoning"] = {"effort": eff}
                if verbosity:
                    payload["text"] = {"verbosity": verbosity}
            if response_format:
                payload["text_format"] = response_format

        if extra:
            payload.update(extra)

        return payload