"""Versioned prompt registry with draft/eval/prod stages and rollback.

Prompt files live under ``prompt-registry/<use_case>/<name>_<version>.json``.
Stage pointers live in ``prompt-registry/index.json``. Stage resolution falls back
``prod -> eval -> highest version`` so local/dev runs always resolve to *something*.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "prompt-registry"
STAGES = ("draft", "eval", "prod")


@dataclass(frozen=True)
class Prompt:
    use_case: str
    version: str
    stage: str
    system: str
    user_template: str
    params: dict[str, Any]

    def render(self, **kwargs: Any) -> str:
        return self.user_template.format(**kwargs)


class PromptRegistry:
    def __init__(self, root: Path | str = DEFAULT_REGISTRY) -> None:
        self.root = Path(root)
        self._index_path = self.root / "index.json"

    # ------------------------------------------------------------------ index
    def _read_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text())

    def _write_index(self, index: dict[str, Any]) -> None:
        self._index_path.write_text(json.dumps(index, indent=2) + "\n")

    def _use_case_pointer(self, index: dict[str, Any], use_case: str) -> dict[str, Any]:
        return index.setdefault(use_case, {s: None for s in STAGES})

    # --------------------------------------------------------------- loaders
    def list_versions(self, use_case: str) -> list[str]:
        d = self.root / use_case
        if not d.is_dir():
            return []
        versions = []
        for f in d.glob("*.json"):
            try:
                versions.append(json.loads(f.read_text())["version"])
            except (json.JSONDecodeError, KeyError):
                continue

        def key(v: str) -> int:
            try:
                return int(v.lstrip("vV"))
            except ValueError:
                return 0

        return sorted(set(versions), key=key)

    def get(self, use_case: str, version: str) -> Prompt:
        d = self.root / use_case
        for f in d.glob("*.json"):
            data = json.loads(f.read_text())
            if data.get("version") == version:
                return self._to_prompt(use_case, "pinned", data)
        raise KeyError(f"prompt {use_case}:{version} not found")

    def resolve(self, use_case: str, stage: str = "prod") -> Prompt:
        index = self._read_index()
        pointer = index.get(use_case, {})
        chosen = pointer.get(stage) or pointer.get("eval") or pointer.get("prod")
        if not chosen:
            versions = self.list_versions(use_case)
            if not versions:
                raise KeyError(f"no prompts registered for use case '{use_case}'")
            chosen = versions[-1]
        return self.get(use_case, chosen)

    def render(self, use_case: str, stage: str = "prod", **kwargs: Any) -> str:
        return self.resolve(use_case, stage).render(**kwargs)

    # -------------------------------------------------------------- lifecycle
    def promote(self, use_case: str, version: str, stage: str) -> Prompt:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        if version not in self.list_versions(use_case):
            raise KeyError(f"cannot promote unknown version {use_case}:{version}")
        index = self._read_index()
        pointer = self._use_case_pointer(index, use_case)
        history = index.setdefault("_history", {}).setdefault(use_case, {}).setdefault(stage, [])
        if pointer.get(stage) and pointer[stage] != version:
            history.append(pointer[stage])
        pointer[stage] = version
        self._write_index(index)
        return self.get(use_case, version)

    def rollback(self, use_case: str, stage: str) -> Prompt:
        index = self._read_index()
        pointer = self._use_case_pointer(index, use_case)
        history = index.setdefault("_history", {}).setdefault(use_case, {}).setdefault(stage, [])
        if not history:
            raise RuntimeError(f"no rollback target saved for {use_case}:{stage}")
        pointer[stage] = history.pop()
        self._write_index(index)
        return self.resolve(use_case, stage)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _to_prompt(use_case: str, stage: str, data: dict[str, Any]) -> Prompt:
        return Prompt(
            use_case=use_case,
            version=data["version"],
            stage=stage,
            system=data.get("system", ""),
            user_template=data.get("user_template", ""),
            params=data.get("params", {}),
        )
