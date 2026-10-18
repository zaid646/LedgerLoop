import json

import pytest

from app.registry import PromptRegistry


@pytest.fixture()
def registry(tmp_path):
    root = tmp_path / "prompt-registry"
    (root / "extraction").mkdir(parents=True)
    (root / "index.json").write_text(
        json.dumps({"extraction": {"draft": None, "eval": "v1", "prod": None}})
    )
    for v in ("v1", "v2"):
        (root / "extraction" / f"extract_{v}.json").write_text(
            json.dumps(
                {
                    "use_case": "extraction",
                    "version": v,
                    "system": f"sys-{v}",
                    "user_template": "text={text}",
                    "params": {},
                }
            )
        )
    return PromptRegistry(root)


def test_resolve_falls_back_to_eval(registry):
    p = registry.resolve("extraction", "prod")
    assert p.version == "v1"
    assert p.render(text="hello") == "text=hello"


def test_list_versions_sorted(registry):
    assert registry.list_versions("extraction") == ["v1", "v2"]


def test_promote_and_rollback(registry):
    registry.promote("extraction", "v2", "prod")
    assert registry.resolve("extraction", "prod").version == "v2"
    # promote v1 over v2, then rollback returns v2
    registry.promote("extraction", "v1", "prod")
    assert registry.resolve("extraction", "prod").version == "v1"
    p = registry.rollback("extraction", "prod")
    assert p.version == "v2"


def test_promote_unknown_version_raises(registry):
    with pytest.raises(KeyError):
        registry.promote("extraction", "v9", "prod")
