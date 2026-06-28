from __future__ import annotations

import uuid
import logging
import threading
from contextlib import contextmanager
from typing import Any, Generator

from .config import settings, PricingConfig

logger = logging.getLogger(__name__)


class Tracer:
    """Langfuse-based tracer for RAG pipeline steps.

    Usage:
        tracer = Tracer(enabled=settings.enabled)
        with tracer.trace_step("retrieve", input={"query": q}) as span:
            results = vector_store.search(q)
            span["output"] = {"results_count": len(results)}
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._langfuse = None
        self._pricing = PricingConfig(settings.pricing_file)
        self._local = threading.local()
        if enabled and settings.langfuse_secret_key:
            self._init_langfuse()

    @property
    def _trace_id(self) -> str:
        if not hasattr(self._local, "trace_id"):
            self._local.trace_id = ""
        return self._local.trace_id

    @_trace_id.setter
    def _trace_id(self, val: str) -> None:
        self._local.trace_id = val

    @property
    def _active_spans(self) -> list[Any]:
        if not hasattr(self._local, "active_spans"):
            self._local.active_spans = []
        return self._local.active_spans

    @_active_spans.setter
    def _active_spans(self, val: list[Any]) -> None:
        self._local.active_spans = val


    def _init_langfuse(self) -> None:
        try:
            from langfuse import Langfuse as LangfuseClient

            self._langfuse = LangfuseClient(
                secret_key=settings.langfuse_secret_key,
                public_key=settings.langfuse_public_key,
                host=settings.langfuse_host,
                release=settings.langfuse_release,
            )
            logger.info("Langfuse initialised: host=%s", settings.langfuse_host)
        except ImportError:
            logger.warning("langfuse package not installed, tracing disabled")
            self.enabled = False

    def get_trace_id(self) -> str:
        if self._trace_id:
            return self._trace_id
        self._trace_id = str(uuid.uuid4())
        return self._trace_id

    @contextmanager
    def trace_step(
        self,
        name: str,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        if not self.enabled or self._langfuse is None:
            span: dict[str, Any] = {"name": name, "input": input or {}, "output": {}, "metadata": metadata or {}}
            self._active_spans.append(span)
            try:
                yield span
            finally:
                self._active_spans.pop()
                if not self._active_spans:
                    self._trace_id = ""
            return

        if not self._active_spans:
            # Root trace
            current = self._langfuse.trace(
                name=name,
                id=self.get_trace_id(),
                input=input,
                metadata=metadata or {},
            )
        else:
            # Sub-span
            parent = self._active_spans[-1]
            current = parent.span(
                name=name,
                input=input,
                metadata=metadata or {},
            )

        self._active_spans.append(current)
        try:
            yield current
        except Exception as exc:
            if hasattr(current, "observation"):
                current.observation(
                    name=f"{name}.error",
                    type="span",
                    level="ERROR",
                    status_message=str(exc),
                )
            raise
        finally:
            self._active_spans.pop()
            
            output_val = getattr(current, "_output", {})
            if not output_val and hasattr(current, "output"):
                output_val = current.output
            if not output_val and isinstance(current, dict):
                output_val = current.get("output", {})

            if hasattr(current, "end"):
                current.end(output=output_val)
            elif hasattr(current, "update"):
                current.update(output=output_val)

            if not self._active_spans:
                self._trace_id = ""

    def capture_generation(
        self,
        span: Any,
        model: str,
        provider: str,
        prompt: str,
        query: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        if not self.enabled or self._langfuse is None:
            return
        cost = self._pricing.get_cost(provider, model, prompt_tokens, completion_tokens)
        span.generation(
            name=f"{span.name}.llm_call",
            model=model,
            model_parameters={"temperature": 0.0, "max_tokens": 1024},
            input=prompt + "\n" + query,
            output=response,
            usage={"prompt": prompt_tokens, "completion": completion_tokens},
            unit="CHARACTERS",
            cost=cost,
        )

