from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class MonitoringSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MONITOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = True
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    langfuse_release: str = "dev"
    otel_service_name: str = "rag-pipeline"
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    pricing_file: str = "pricing.json"
    baseline_dir: str = "data/monitoring"
    query_timeout_seconds: int = 120
    async_telemetry: bool = True
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_seconds: float = 30.0
    otel_export_interval_ms: int = 10000
    otel_export_timeout_ms: int = 5000
    max_queue_size: int = 10000

    @field_validator("langfuse_secret_key", "langfuse_public_key", mode="before")
    @classmethod
    def validate_and_strip_keys(cls, v: Any) -> str:
        if not isinstance(v, str):
            return ""
        stripped = v.strip()
        if stripped and not (
            stripped.startswith("sk-lf-")
            or stripped.startswith("pk-lf-")
            or stripped.startswith("sk-")
            or stripped.startswith("pk-")
        ):
            # Do NOT log the key value — even a partial prefix reduces brute-force space.
            logging.getLogger(__name__).warning(
                "Monitoring key format might be incorrect (expected sk-lf- or pk-lf- prefix)"
            )
        return stripped


class PricingConfig:
    _cache: dict[str, dict[str, Any]] = {}

    def __init__(self, pricing_path: str | Path = "pricing.json") -> None:
        self.path = Path(pricing_path)
        if not self.path.exists():
            self.path = Path(__file__).parent.parent / pricing_path

        self.abs_path = str(self.path.resolve())
        self._last_loaded = 0.0
        self._data: dict[str, Any] = {}
        self._load_data()

    def _load_data(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
            if mtime > self._last_loaded:
                with open(self.path) as f:
                    self._data = json.load(f)
                self._last_loaded = mtime
                PricingConfig._cache[self.abs_path] = self._data
            else:
                self._data = PricingConfig._cache.get(self.abs_path, self._data)
        except Exception as e:
            logger.warning("Failed to load pricing config: %s", e)
            if self.abs_path in PricingConfig._cache:
                self._data = PricingConfig._cache[self.abs_path]

    def get_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        self._load_data()
        provider_data = self._data.get(provider.lower(), {})
        model_data = provider_data.get(model)

        if not model_data:
            import re

            for key, val in provider_data.items():
                # Escape all regex metacharacters in literal segments, then restore * wildcards.
                # This prevents ReDoS if pricing.json contains crafted model-name patterns.
                safe_pattern = ".*".join(re.escape(seg) for seg in key.split("*"))
                if re.match(f"^{safe_pattern}$", model) or key in model:
                    model_data = val
                    break

        if not model_data:
            model_data = {"input": 0.0, "output": 0.0}

        input_price = model_data.get("input", 0.0) / 1000
        output_price = model_data.get("output", 0.0) / 1000
        return (prompt_tokens * input_price) + (completion_tokens * output_price)


settings = MonitoringSettings()
