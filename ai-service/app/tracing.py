"""Langfuse tracing wrapper.

Never breaks a request: if Langfuse is unconfigured or its SDK/API changes, all
methods degrade to no-ops. Traces capture prompt version, model, input/output,
latency and token usage so quality/cost are auditable per document.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .config import get_settings


class Tracer:
    def __init__(self) -> None:
        s = get_settings()
        self.enabled = s.langfuse_enabled
        self.host = s.langfuse_host
        self._client: Any = None
        if self.enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=s.langfuse_public_key,
                    secret_key=s.langfuse_secret_key,
                    host=s.langfuse_host,
                )
            except Exception:  # noqa: BLE001 - observability must never be fatal
                self.enabled = False
                self._client = None

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[Any]:
        span = None
        if self.enabled and self._client is not None:
            try:
                span = self._client.start_span(name=name, **fields)
                span.__enter__()
            except Exception:  # noqa: BLE001
                span = None
        try:
            yield span
        finally:
            if span is not None:
                with contextlib.suppress(Exception):
                    span.__exit__(None, None, None)

    def generation(
        self,
        name: str,
        model: str,
        input: Any,
        output: Any,
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not (self.enabled and self._client is not None):
            return
        client = self._client
        with contextlib.suppress(Exception):
            if hasattr(client, "start_observation"):  # Langfuse SDK >=3 (OTEL)
                gen = client.start_observation(
                    name=name,
                    as_type="generation",
                    model=model,
                    input=input,
                    output=output,
                    metadata=metadata or {},
                )
                gen.end()
            elif hasattr(client, "start_generation"):  # early v3 alias
                gen = client.start_generation(
                    name=name,
                    model=model,
                    input=input,
                    output=output,
                    usage=usage or {},
                    metadata=metadata or {},
                )
                gen.end()
            else:  # Langfuse SDK v2
                client.trace(name=name, metadata=metadata or {}).generation(
                    name=name,
                    model=model,
                    input=input,
                    output=output,
                    usage=usage or {},
                )
            client.flush()

    def score(
        self,
        name: str,
        value: float,
        comment: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach a numeric quality score to Langfuse (no-op when disabled)."""
        if not (self.enabled and self._client is not None):
            return
        client = self._client
        with contextlib.suppress(Exception):
            if hasattr(client, "create_score"):
                client.create_score(
                    name=name,
                    value=float(value),
                    comment=comment,
                    metadata=metadata or {},
                )
            elif hasattr(client, "score"):
                client.score(
                    name=name,
                    value=float(value),
                    comment=comment,
                    metadata=metadata or {},
                )
            client.flush()

    def flush(self) -> None:
        if self.enabled and self._client is not None:
            with contextlib.suppress(Exception):
                self._client.flush()


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
