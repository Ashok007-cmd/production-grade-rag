from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class PromptVersion:
    hash: str
    prompt: str
    name: str
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptRegistry:
    """Version-controlled prompt template registry.

    Detects prompt changes via SHA256 hash comparison, enabling
    CI regression gating when prompt templates are modified.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._versions: dict[str, list[PromptVersion]] = {}
        self._current: dict[str, str] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def register(
        self,
        name: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        version = PromptVersion(
            hash=prompt_hash,
            prompt=prompt,
            name=name,
            timestamp=datetime.now(UTC).isoformat(),
            metadata=metadata or {},
        )
        if name not in self._versions:
            self._versions[name] = []
        self._versions[name].append(version)
        self._current[name] = prompt_hash
        self._save()
        return prompt_hash

    def current_hash(self, name: str) -> str | None:
        return self._current.get(name)

    def detect_change(self, name: str, new_prompt: str) -> bool:
        current = self.current_hash(name)
        if current is None:
            return True
        return current != hashlib.sha256(new_prompt.encode()).hexdigest()

    def get_versions(self, name: str) -> list[PromptVersion]:
        return self._versions.get(name, [])

    def _save(self) -> None:
        if not self._persist_path:
            return
        import os

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: [asdict(v) for v in versions] for name, versions in self._versions.items()}
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(self._persist_path, flags, 0o600)
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        assert self._persist_path is not None
        with open(self._persist_path) as f:
            data = json.load(f)
        for name, versions_data in data.items():
            self._versions[name] = [PromptVersion(**v) for v in versions_data]
            if versions_data:
                self._current[name] = versions_data[-1]["hash"]
