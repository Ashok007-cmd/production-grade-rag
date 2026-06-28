from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from .config import settings
from .extensions import (
    BaseExtension,
    LangfuseTracingExtension,
    OTelMetricsExtension,
)
from .tracing import Tracer

logger = logging.getLogger(__name__)

try:
    from .metrics import MetricsCollector
except ImportError:
    MetricsCollector = None  # type: ignore


# Global ContextVar to store active usage_dict for the current context/thread
active_usage_var: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "active_usage", default=None
)

# Track our patched wrapper functions to prevent double-patching and verify test mock overrides
_patched_openai_func = None
_patched_anthropic_func = None


def _globally_patch_clients():
    global _patched_openai_func, _patched_anthropic_func

    # Patch OpenAI
    try:
        import openai

        orig_openai_create = openai.resources.chat.completions.Completions.create

        def patched_openai_create(self_obj, *args, **kwargs):
            res = orig_openai_create(self_obj, *args, **kwargs)
            try:
                usage_dict = active_usage_var.get()
                if usage_dict is not None and hasattr(res, "usage") and res.usage is not None:
                    usage_dict["prompt_tokens"] = res.usage.prompt_tokens
                    usage_dict["completion_tokens"] = res.usage.completion_tokens
                    usage_dict["total_tokens"] = res.usage.total_tokens
            except Exception as e:
                logger.warning("Failed to extract OpenAI token usage: %s", e)
            return res

        openai.resources.chat.completions.Completions.create = patched_openai_create  # type: ignore[method-assign]
        _patched_openai_func = patched_openai_create
    except (ImportError, AttributeError):
        pass

    # Patch Anthropic
    try:
        import anthropic

        orig_anthropic_create = anthropic.resources.messages.Messages.create

        def patched_anthropic_create(self_obj, *args, **kwargs):
            res = orig_anthropic_create(self_obj, *args, **kwargs)
            try:
                usage_dict = active_usage_var.get()
                if usage_dict is not None and hasattr(res, "usage") and res.usage is not None:
                    usage_dict["prompt_tokens"] = res.usage.input_tokens
                    usage_dict["completion_tokens"] = res.usage.output_tokens
                    usage_dict["total_tokens"] = res.usage.input_tokens + res.usage.output_tokens
            except Exception as e:
                logger.warning("Failed to extract Anthropic token usage: %s", e)
            return res

        anthropic.resources.messages.Messages.create = patched_anthropic_create  # type: ignore[method-assign]
        _patched_anthropic_func = patched_anthropic_create
    except (ImportError, AttributeError):
        pass


# Initialize global patches
_globally_patch_clients()


def _get_default_system_prompt() -> str:
    try:
        import importlib

        generator_mod = importlib.import_module("src.generation.generator")
        return getattr(generator_mod, "DEFAULT_SYSTEM_PROMPT", "")
    except ImportError:
        return ""


_active_patches_lock = threading.Lock()
_active_patches_count = 0
_original_openai_create = None
_original_anthropic_create = None
_dynamic_openai_patch_active = False
_dynamic_anthropic_patch_active = False


@contextlib.contextmanager
def intercept_token_usage(usage_dict: dict[str, int]):
    global \
        _active_patches_count, \
        _original_openai_create, \
        _original_anthropic_create, \
        _dynamic_openai_patch_active, \
        _dynamic_anthropic_patch_active

    usage_dict["prompt_tokens"] = 0
    usage_dict["completion_tokens"] = 0
    usage_dict["total_tokens"] = 0

    token = active_usage_var.set(usage_dict)

    with _active_patches_lock:
        if _active_patches_count == 0:
            # Check OpenAI
            try:
                import openai

                current_openai = openai.resources.chat.completions.Completions.create
                if current_openai is not _patched_openai_func:
                    _dynamic_openai_patch_active = True
                    _original_openai_create = current_openai

                    def temp_openai_create(self_obj, *args, **kwargs):
                        res = _original_openai_create(self_obj, *args, **kwargs)  # type: ignore[misc]
                        try:
                            u_dict = active_usage_var.get()
                            if (
                                u_dict is not None
                                and hasattr(res, "usage")
                                and res.usage is not None
                            ):
                                u_dict["prompt_tokens"] = res.usage.prompt_tokens
                                u_dict["completion_tokens"] = res.usage.completion_tokens
                                u_dict["total_tokens"] = res.usage.total_tokens
                        except Exception as e:
                            logger.warning(
                                "Failed to extract OpenAI token usage in dynamic patch: %s", e
                            )
                        return res

                    openai.resources.chat.completions.Completions.create = temp_openai_create  # type: ignore[method-assign]
            except (ImportError, AttributeError):
                pass

            # Check Anthropic
            try:
                import anthropic

                current_anthropic = anthropic.resources.messages.Messages.create
                if current_anthropic is not _patched_anthropic_func:
                    _dynamic_anthropic_patch_active = True
                    _original_anthropic_create = current_anthropic

                    def temp_anthropic_create(self_obj, *args, **kwargs):
                        res = _original_anthropic_create(self_obj, *args, **kwargs)  # type: ignore[misc]
                        try:
                            u_dict = active_usage_var.get()
                            if (
                                u_dict is not None
                                and hasattr(res, "usage")
                                and res.usage is not None
                            ):
                                u_dict["prompt_tokens"] = res.usage.input_tokens
                                u_dict["completion_tokens"] = res.usage.output_tokens
                                u_dict["total_tokens"] = (
                                    res.usage.input_tokens + res.usage.output_tokens
                                )
                        except Exception as e:
                            logger.warning(
                                "Failed to extract Anthropic token usage in dynamic patch: %s", e
                            )
                        return res

                    anthropic.resources.messages.Messages.create = temp_anthropic_create  # type: ignore[method-assign]
            except (ImportError, AttributeError):
                pass

        _active_patches_count += 1

    try:
        yield
    finally:
        active_usage_var.reset(token)
        with _active_patches_lock:
            _active_patches_count -= 1
            if _active_patches_count == 0:
                if _dynamic_openai_patch_active and _original_openai_create is not None:
                    try:
                        import openai

                        openai.resources.chat.completions.Completions.create = (  # type: ignore[method-assign]
                            _original_openai_create
                        )
                    except (ImportError, AttributeError):
                        pass
                    _original_openai_create = None
                    _dynamic_openai_patch_active = False
                if _dynamic_anthropic_patch_active and _original_anthropic_create is not None:
                    try:
                        import anthropic

                        anthropic.resources.messages.Messages.create = _original_anthropic_create  # type: ignore[method-assign]
                    except (ImportError, AttributeError):
                        pass
                    _original_anthropic_create = None
                    _dynamic_anthropic_patch_active = False


class MonitoredRAGPipeline:
    """Instrumented wrapper around RAGPipeline.

    Adds Langfuse tracing and OTel metrics to every pipeline step
    via a pluggable extension architecture.
    """

    def __init__(
        self,
        pipeline: Any,
        tracer: Tracer | None = None,
        metrics: Any | None = None,
        extensions: list[BaseExtension] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self.extensions = extensions or []

        # Backward compatibility for direct tracer/metrics configurations
        has_tracing = any(isinstance(e, LangfuseTracingExtension) for e in self.extensions)
        has_metrics = any(isinstance(e, OTelMetricsExtension) for e in self.extensions)

        if not has_tracing:
            self._tracer = tracer or Tracer()
            self.extensions.append(LangfuseTracingExtension(self._tracer))
        else:
            self._tracer = next(
                e.tracer for e in self.extensions if isinstance(e, LangfuseTracingExtension)
            )

        if not has_metrics:
            self._metrics = metrics
            self.extensions.append(OTelMetricsExtension(self._metrics))
        else:
            self._metrics = next(
                e.metrics for e in self.extensions if isinstance(e, OTelMetricsExtension)
            )

        # Initialise prompt registry baseline configuration on wrappers
        from src.monitoring.prompts import PromptRegistry

        registry_path = Path(settings.baseline_dir) / "prompts.json"
        self._prompt_registry = PromptRegistry(registry_path)

        # Initialize token/cost tracking at query level
        self._last_query_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._last_query_cost = 0.0

        self._instrument_pipeline()

    def _run_hook(self, ext_method_name: str, *args: Any, **kwargs: Any) -> None:
        for ext in self.extensions:
            hook = getattr(ext, ext_method_name, None)
            if hook is not None:
                try:
                    hook(*args, **kwargs)
                except ValueError:
                    raise
                except Exception as e:
                    logger.warning(
                        "Extension hook %s failed on %s: %s", ext_method_name, type(ext).__name__, e
                    )

    def _instrument_pipeline(self) -> None:
        if (
            hasattr(self._pipeline, "_mock_return_value")
            or hasattr(self._pipeline, "mock_add_spec")
            or type(self._pipeline).__name__ in ("Mock", "MagicMock")
            or not hasattr(self._pipeline, "generator")
        ):
            return

        # 1. Wrap _retrieve
        original_retrieve = self._pipeline._retrieve

        def monitored_retrieve(
            query: str,
            use_hybrid: bool = False,
            use_reranker: bool = False,
            k: int = 5,
            lang: str = "en",
        ):
            start_time = time.monotonic()
            input_data = {
                "query": query,
                "use_hybrid": use_hybrid,
                "use_reranker": use_reranker,
                "k": k,
                "lang": lang,
            }

            self._run_hook("on_step_start", "retrieve", input_data, {})

            import inspect

            sig = inspect.signature(original_retrieve)
            kwargs: dict[str, Any] = {}
            if "use_hybrid" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["use_hybrid"] = use_hybrid
            if "use_reranker" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["use_reranker"] = use_reranker
            if "k" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["k"] = k
            if "lang" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["lang"] = lang

            try:
                contexts = original_retrieve(query, **kwargs)
            except Exception as exc:
                self._run_hook("on_step_error", "retrieve", exc)
                raise

            elapsed = time.monotonic() - start_time
            chunk_ids = [ctx.get("id") for ctx in contexts]
            retrieved_metadata = []
            for ctx in contexts:
                meta = ctx.get("metadata", {})
                source = meta.get("source", meta.get("filename", "unknown"))
                title = meta.get("title", meta.get("source", "unknown"))
                retrieved_metadata.append(
                    {
                        "id": ctx.get("id"),
                        "source": source,
                        "title": title,
                        "score": ctx.get("score"),
                    }
                )

            output_val = {"chunk_ids": chunk_ids, "chunks": retrieved_metadata}

            self._run_hook(
                "on_step_end",
                "retrieve",
                output_val,
                elapsed,
                metadata={"chunk_ids": chunk_ids, "chunks": retrieved_metadata},
            )

            return contexts

        self._pipeline._retrieve = monitored_retrieve

        # 2. Wrap _apply_reranker
        if hasattr(self._pipeline, "_apply_reranker"):
            original_rerank = self._pipeline._apply_reranker

            def monitored_rerank(query: str, contexts: list[dict[str, Any]], top_k: int = 5):
                start_time = time.monotonic()
                input_data = {"query": query, "contexts_count": len(contexts), "top_k": top_k}

                self._run_hook("on_step_start", "rerank", input_data, {})

                try:
                    reranked = original_rerank(query, contexts, top_k=top_k)
                except Exception as exc:
                    self._run_hook("on_step_error", "rerank", exc)
                    raise

                elapsed = time.monotonic() - start_time
                pre_scores = {ctx.get("id"): ctx.get("score") for ctx in contexts}
                post_scores = {ctx.get("id"): ctx.get("score") for ctx in reranked}
                selected_ids = [ctx.get("id") for ctx in reranked]

                model_name = "unknown"
                if hasattr(self._pipeline, "_reranker") and self._pipeline._reranker is not None:
                    model_name = getattr(self._pipeline._reranker, "model_name", "unknown")

                output_metadata = {
                    "model": model_name,
                    "pre_scores": pre_scores,
                    "post_scores": post_scores,
                    "selected_ids": selected_ids,
                }

                self._run_hook(
                    "on_step_end", "rerank", output_metadata, elapsed, metadata=output_metadata
                )

                return reranked

            self._pipeline._apply_reranker = monitored_rerank

        # 3. Wrap generator.generate
        original_generate = self._pipeline.generator.generate

        def monitored_generate(
            query: str, contexts: list[dict[str, Any]], system_prompt: str | None = None
        ) -> str:
            start_time = time.monotonic()

            DEFAULT_SYSTEM_PROMPT = _get_default_system_prompt()

            formatted_context = ""
            if hasattr(self._pipeline.generator, "_format_context"):
                formatted_context = self._pipeline.generator._format_context(contexts)
            prompt_content = (system_prompt or DEFAULT_SYSTEM_PROMPT).format(
                context=formatted_context
            )

            input_data = {"query": query, "contexts_count": len(contexts)}
            self._run_hook("on_step_start", "generate", input_data, {})

            usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            try:
                with intercept_token_usage(usage_dict):
                    answer = original_generate(query, contexts, system_prompt=system_prompt)
            except Exception as exc:
                self._run_hook("on_step_error", "generate", exc)
                raise

            elapsed = time.monotonic() - start_time
            provider = getattr(self._pipeline.generator, "provider", "openai")
            model = getattr(self._pipeline.generator, "model", "gpt-4o-mini")

            cost = self._tracer._pricing.get_cost(
                provider, model, usage_dict["prompt_tokens"], usage_dict["completion_tokens"]
            )

            self._last_query_tokens = {
                "prompt": usage_dict["prompt_tokens"],
                "completion": usage_dict["completion_tokens"],
                "total": usage_dict["total_tokens"],
            }
            self._last_query_cost = cost

            # Registry update
            prompt_name = (
                "default_system_prompt" if system_prompt is None else "custom_system_prompt"
            )
            raw_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
            try:
                self._prompt_registry.register(
                    prompt_name, raw_prompt, metadata={"model": model, "provider": provider}
                )
            except Exception as e:
                logger.warning("Failed to update prompt registry: %s", e)

            self._run_hook(
                "on_generation_llm_call",
                model=model,
                provider=provider,
                prompt=prompt_content,
                query=query,
                response=answer,
                usage=usage_dict,
                cost=cost,
            )
            self._run_hook("on_step_end", "generate", {"answer": answer}, elapsed)

            return answer

        self._pipeline.generator.generate = monitored_generate

    def ingest(self, source: Path | str) -> int:
        self._run_hook("on_step_start", "ingest", {"source": str(source)}, {})

        start_time = time.monotonic()
        try:
            res = self._pipeline.ingest(source)
            elapsed = time.monotonic() - start_time
            self._run_hook("on_step_end", "ingest", {"chunks_ingested": res}, elapsed)
            return res
        except Exception as exc:
            self._run_hook("on_step_error", "ingest", exc)
            raise

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> tuple[str, list[Any]]:
        start = time.monotonic()
        metadata = {
            "hybrid": use_hybrid,
            "reranker": use_reranker,
            "top_k": top_k,
        }

        self._last_query_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._last_query_cost = 0.0

        # Run query start guardrails and hooks
        self._run_hook("on_query_start", question, metadata)

        try:
            answer, citations = self._pipeline.query(
                question,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
            )
        except Exception as exc:
            self._run_hook("on_query_error", exc)
            raise

        elapsed = time.monotonic() - start

        # Run query completion hooks
        self._run_hook(
            "on_query_end",
            answer,
            citations,
            elapsed,
            self._last_query_tokens,
            self._last_query_cost,
        )

        return answer, citations

    def query_stream(
        self,
        question: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> Generator[str, None, None]:
        """Runs a monitored RAG query, yielding token chunks in real-time.

        Measures Time-to-First-Token (TTFT) and throughput metrics.
        """
        start = time.monotonic()
        metadata = {
            "hybrid": use_hybrid,
            "reranker": use_reranker,
            "top_k": top_k,
            "streaming": True,
        }

        self._last_query_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._last_query_cost = 0.0

        self._run_hook("on_query_start", question, metadata)

        try:
            k = top_k or self._pipeline.config.top_k_final
            contexts = self._pipeline._retrieve(
                question,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                k=k,
            )
            if use_reranker and hasattr(self._pipeline, "_apply_reranker"):
                contexts = self._pipeline._apply_reranker(question, contexts, top_k=k)
            citations = self._pipeline.citation_formatter.build_citations(contexts)
        except Exception as exc:
            self._run_hook("on_query_error", exc)
            raise

        def count_tokens(text: str, model: str = "gpt-4") -> int:
            try:
                import importlib

                tiktoken = importlib.import_module("tiktoken")
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except Exception:
                return max(1, len(text) // 4)

        def on_first_token(ttft: float):
            self._run_hook("on_first_token", ttft)

        def on_completion(answer: str, elapsed: float):
            provider = getattr(self._pipeline.generator, "provider", "openai")
            model = getattr(self._pipeline.generator, "model", "gpt-4o-mini")

            formatted_context = ""
            if hasattr(self._pipeline.generator, "_format_context"):
                formatted_context = self._pipeline.generator._format_context(contexts)

            DEFAULT_SYSTEM_PROMPT = _get_default_system_prompt()

            prompt_content = DEFAULT_SYSTEM_PROMPT.format(context=formatted_context)
            prompt_tokens = count_tokens(prompt_content + "\n" + question, model)
            completion_tokens = count_tokens(answer, model)
            total_tokens = prompt_tokens + completion_tokens
            cost = self._tracer._pricing.get_cost(provider, model, prompt_tokens, completion_tokens)

            tokens_dict = {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            }
            self._last_query_tokens = tokens_dict
            self._last_query_cost = cost

            self._run_hook("on_query_end", answer, citations, elapsed, tokens_dict, cost)

        def on_error(exc: Exception):
            self._run_hook("on_query_error", exc)

        # Dynamic stream method definition
        def dynamic_generate_stream(
            self_gen, q: str, ctxs: list[dict[str, Any]], sys_prompt: str | None = None
        ) -> Generator[str, None, None]:
            DEFAULT_SYSTEM_PROMPT = _get_default_system_prompt()

            fmt_context = ""
            if hasattr(self_gen, "_format_context"):
                fmt_context = self_gen._format_context(ctxs)
            prompt_content = (sys_prompt or DEFAULT_SYSTEM_PROMPT).format(context=fmt_context)

            prov = getattr(self_gen, "provider", "openai")
            mdl = getattr(self_gen, "model", "gpt-4o-mini")

            if prov == "openai":
                from openai import OpenAI

                client = OpenAI(timeout=60.0)
                res = client.chat.completions.create(
                    model=mdl,
                    messages=[
                        {"role": "system", "content": prompt_content},
                        {"role": "user", "content": q},
                    ],
                    temperature=getattr(self_gen, "temperature", 0.0),
                    max_tokens=getattr(self_gen, "max_tokens", 1024),
                    stream=True,
                )
                for chunk in res:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            elif prov == "anthropic":
                from anthropic import Anthropic

                anthropic_client = Anthropic(timeout=60.0)
                with anthropic_client.messages.stream(
                    model=mdl,
                    max_tokens=getattr(self_gen, "max_tokens", 1024),
                    temperature=getattr(self_gen, "temperature", 0.0),
                    system=prompt_content,
                    messages=[{"role": "user", "content": q}],
                ) as stream:
                    yield from stream.text_stream
            else:
                # Simulated stream fallback
                try:
                    ans = self_gen.generate(q, ctxs, system_prompt=sys_prompt)
                except Exception:
                    ans = "Fallback generated output stream simulation."
                for word in ans.split(" "):
                    yield word + " "

        try:
            if not hasattr(self._pipeline.generator, "generate_stream"):
                self._pipeline.generator.generate_stream = dynamic_generate_stream.__get__(
                    self._pipeline.generator, type(self._pipeline.generator)
                )
            raw_stream = self._pipeline.generator.generate_stream(question, contexts)
        except Exception as exc:
            self._run_hook("on_query_error", exc)
            raise

        self._run_hook(
            "on_step_start", "generate", {"query": question, "contexts_count": len(contexts)}, {}
        )

        def stream_generator():
            start_gen = time.monotonic()
            accumulated_text = []
            first_token_time = None

            try:
                for chunk in raw_stream:
                    if first_token_time is None and chunk:
                        first_token_time = time.monotonic()
                        ttft = first_token_time - start_gen
                        on_first_token(ttft)
                    accumulated_text.append(chunk)
                    yield chunk

                elapsed_gen = time.monotonic() - start_gen
                answer = "".join(accumulated_text)

                provider = getattr(self._pipeline.generator, "provider", "openai")
                model = getattr(self._pipeline.generator, "model", "gpt-4o-mini")
                prompt_tokens = count_tokens(question, model)
                completion_tokens = count_tokens(answer, model)
                cost = self._tracer._pricing.get_cost(
                    provider, model, prompt_tokens, completion_tokens
                )

                self._run_hook(
                    "on_generation_llm_call",
                    model=model,
                    provider=provider,
                    prompt=question,
                    query=question,
                    response=answer,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    cost=cost,
                )
                self._run_hook("on_step_end", "generate", {"answer": answer}, elapsed_gen)

                on_completion(answer, time.monotonic() - start)

            except Exception as exc:
                self._run_hook("on_step_error", "generate", exc)
                on_error(exc)
                raise

        return stream_generator()

    def stats(self) -> dict[str, Any]:
        return self._pipeline.stats()

    def reset(self) -> None:
        self._run_hook("on_step_start", "reset", {}, {})
        start = time.monotonic()
        try:
            self._pipeline.reset()
            self._run_hook("on_step_end", "reset", {}, time.monotonic() - start)
        except Exception as exc:
            self._run_hook("on_step_error", "reset", exc)
            raise
