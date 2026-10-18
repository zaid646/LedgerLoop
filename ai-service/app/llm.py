"""OpenAI-compatible async client for the self-hosted vLLM endpoint.

Supports schema-guided JSON output. vLLM accepts OpenAI-style
``response_format={"type": "json_schema", ...}``; as a fallback we retry with
vLLM's native ``guided_json`` extra body, then with plain ``json_object``.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .config import get_settings


@dataclass
class LLMResult:
    content: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


def gpu_rate_usd_hr() -> float:
    """Hourly GPU rate for the active vLLM session (0 => cost unknown)."""
    try:
        return float(os.getenv("GPU_RATE_USD_HR", "0") or 0)
    except ValueError:
        return 0.0


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(
            base_url=base_url or s.vllm_base_url,
            api_key=api_key or s.vllm_api_key,
            timeout=90.0,
        )

    @staticmethod
    def _cost(latency_ms: float) -> float:
        # Dedicated-GPU model: cost == rate * wall-clock occupied (seconds).
        return gpu_rate_usd_hr() * (latency_ms / 1000.0) / 3600.0

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> LLMResult:
        s = get_settings()
        model = model or s.vllm_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": False},
            }

        start = time.perf_counter()
        resp = await self._call_with_fallback(kwargs, schema, model)
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return LLMResult(
            content=choice.message.content or "",
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=self._cost(latency_ms),
        )

    async def _call_with_fallback(
        self, kwargs: dict[str, Any], schema: dict[str, Any] | None, model: str
    ) -> Any:
        try:
            return await self._client.chat.completions.create(**kwargs)
        except Exception:
            if schema is None:
                raise
        # Fallback 1: vLLM native guided decoding
        alt = dict(kwargs)
        alt.pop("response_format", None)
        alt["extra_body"] = {"guided_json": schema}
        try:
            return await self._client.chat.completions.create(**alt)
        except Exception:
            pass
        # Fallback 2: plain JSON object mode
        alt2 = dict(kwargs)
        alt2["response_format"] = {"type": "json_object"}
        return await self._client.chat.completions.create(**alt2)

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> tuple[dict[str, Any], LLMResult]:
        result = await self.complete(
            messages, model=model, schema=schema, temperature=temperature, max_tokens=max_tokens
        )
        return _loads_lenient(result.content), result

    async def list_models(self) -> list[str]:
        resp = await self._client.models.list()
        return [m.id for m in resp.data]


def _loads_lenient(content: str) -> dict[str, Any]:
    """Parse JSON, tolerating markdown fences / leading prose."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.removesuffix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# ------------------------------------------------------------------ sync shims
def complete_json_sync(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], LLMResult]:
    """Blocking helper for notebooks / scripts."""
    return asyncio.run(LLMClient().complete_json(*args, **kwargs))
