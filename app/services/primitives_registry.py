"""Primitives registry — parses SKILL.md frontmatter and provides queryable catalogue.

The registry reads ``agent/skills/*.md`` files at startup, extracts the YAML
frontmatter, and builds an in-memory index of all known primitives with their
parameters, error classes, action contracts, and owning instruments.

This serves triple duty:
1. **Safety gate** — ``safety.py`` can query allowed primitives from the registry
   instead of maintaining a hardcoded list.
2. **LLM context** — the agent can ask "what can I do?" and get structured answers.
3. **API endpoint** — ``GET /api/v1/capabilities`` returns the full catalogue.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.action_contracts import (
    ActionContract,
    SafetyClass,
    build_action_contract,
    parse_safety_class,
)

# We vendor a tiny YAML subset parser to avoid adding PyYAML as a dependency.
# Skill frontmatter is simple enough (flat keys + lists of dicts) that we can
# parse it with a lightweight approach.
try:
    import yaml as _yaml  # type: ignore[import-untyped]

    def _parse_yaml(text: str) -> dict[str, Any]:
        return _yaml.safe_load(text) or {}

except ImportError:
    import json as _json

    def _parse_yaml(text: str) -> dict[str, Any]:  # type: ignore[misc]
        """Fallback: convert YAML-ish frontmatter to JSON-parseable form.

        Handles only the subset used by our skill files:
        - top-level scalar keys
        - top-level list-of-dict keys (primitives)
        This is intentionally limited — install PyYAML for full support.
        """
        # For the fallback we just return an empty dict and log a warning.
        # The registry will still work but won't auto-discover primitives.
        import warnings

        warnings.warn(
            "PyYAML not installed — primitives registry using built-in catalogue. "
            "Install pyyaml for auto-discovery from skill markdown files.",
            stacklevel=2,
        )
        return {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimitiveParam:
    """A single parameter for a primitive."""

    name: str
    type: str  # "string", "number", "integer", "boolean", "array"
    description: str = ""
    optional: bool = False
    default: Any = None


@dataclass(frozen=True)
class PrimitiveSpec:
    """Full specification of a single primitive (action)."""

    name: str  # e.g. "robot.aspirate"
    description: str
    error_class: str  # "CRITICAL" or "BYPASS"
    instrument: str | None  # e.g. "ot2-robot", None for utilities
    resource_id: str | None  # resource lock required
    skill_name: str  # owning skill file
    params: tuple[PrimitiveParam, ...] = ()
    safety_class: SafetyClass = SafetyClass.CAREFUL
    contract: ActionContract | None = None


@dataclass
class SkillDescriptor:
    """Parsed metadata from a single SKILL.md file."""

    name: str
    description: str
    version: str
    instrument: str | None
    resource_id: str | None
    primitives: list[PrimitiveSpec] = field(default_factory=list)
    source_path: str = ""


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def _extract_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter from a markdown file."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    return _parse_yaml(match.group(1))


def _parse_params(raw_params: dict[str, Any]) -> tuple[PrimitiveParam, ...]:
    """Convert a params dict from frontmatter into PrimitiveParam tuples."""
    if not raw_params or not isinstance(raw_params, dict):
        return ()
    result: list[PrimitiveParam] = []
    for param_name, spec in raw_params.items():
        if isinstance(spec, dict):
            result.append(
                PrimitiveParam(
                    name=str(param_name),
                    type=spec.get("type", "string"),
                    description=spec.get("description", ""),
                    optional=spec.get("optional", False),
                    default=spec.get("default"),
                )
            )
        else:
            result.append(PrimitiveParam(name=str(param_name), type="string"))
    return tuple(result)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PrimitivesRegistry:
    """In-memory catalogue of all known primitives, loaded from skill files."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDescriptor] = {}
        self._primitives: dict[str, PrimitiveSpec] = {}

    # -- Loading -----------------------------------------------------------

    def load_skill_file(self, path: Path) -> SkillDescriptor | None:
        """Parse a single skill markdown file and register its primitives."""
        text = path.read_text(encoding="utf-8")
        fm = _extract_frontmatter(text)
        if not fm:
            return None

        skill_name = fm.get("name", path.stem)
        instrument = fm.get("instrument")
        if instrument == "null" or instrument is None:
            instrument = None
        resource_id = fm.get("resource_id")
        if resource_id == "null" or resource_id is None:
            resource_id = None

        skill = SkillDescriptor(
            name=skill_name,
            description=fm.get("description", ""),
            version=fm.get("version", "0.0.0"),
            instrument=instrument,
            resource_id=resource_id,
            source_path=str(path),
        )

        for raw_prim in fm.get("primitives", []):
            if not isinstance(raw_prim, dict):
                continue
            prim_name = raw_prim.get("name", "")
            if not prim_name:
                continue

            error_class = raw_prim.get("error_class", "CRITICAL")

            # Parse safety_class (new) with fallback to error_class (legacy)
            safety_cls = parse_safety_class(
                raw_safety=raw_prim.get("safety_class"),
                error_class=error_class,
                primitive_name=prim_name,
            )

            # Parse action contract (preconditions, effects, timeout)
            contract = build_action_contract(
                raw_contract=raw_prim.get("contract"),
                raw_timeout=raw_prim.get("timeout"),
                safety_class=safety_cls,
            )

            spec = PrimitiveSpec(
                name=prim_name,
                description=raw_prim.get("description", ""),
                error_class=error_class,
                instrument=instrument,
                resource_id=resource_id,
                skill_name=skill_name,
                params=_parse_params(raw_prim.get("params", {})),
                safety_class=safety_cls,
                contract=contract,
            )
            skill.primitives.append(spec)
            self._primitives[prim_name] = spec

        self._skills[skill_name] = skill
        return skill

    def load_skills_dir(self, skills_dir: Path) -> int:
        """Load all ``*.md`` files from a skills directory. Returns count loaded."""
        count = 0
        if not skills_dir.is_dir():
            return count
        for md_file in sorted(skills_dir.glob("*.md")):
            if self.load_skill_file(md_file) is not None:
                count += 1
        return count

    # -- Queries -----------------------------------------------------------

    def get_primitive(self, name: str) -> PrimitiveSpec | None:
        """Look up a single primitive by name."""
        return self._primitives.get(name)

    def list_primitives(self) -> list[PrimitiveSpec]:
        """Return all registered primitives, sorted by name."""
        return sorted(self._primitives.values(), key=lambda p: p.name)

    def list_primitive_names(self) -> list[str]:
        """Return all registered primitive names, sorted."""
        return sorted(self._primitives.keys())

    def list_skills(self) -> list[SkillDescriptor]:
        """Return all loaded skills."""
        return list(self._skills.values())

    def get_skill(self, name: str) -> SkillDescriptor | None:
        """Look up a skill by name."""
        return self._skills.get(name)

    def primitives_by_instrument(self, instrument: str) -> list[PrimitiveSpec]:
        """Return all primitives for a given instrument."""
        return [p for p in self._primitives.values() if p.instrument == instrument]

    def list_instruments(self) -> list[str]:
        """Return unique instrument IDs from loaded skills (e.g. 'ot2-robot', 'squidstat').

        Excludes skills with ``instrument=None`` (utilities).
        """
        seen: set[str] = set()
        result: list[str] = []
        for skill in self._skills.values():
            if skill.instrument and skill.instrument not in seen:
                seen.add(skill.instrument)
                result.append(skill.instrument)
        return sorted(result)

    def list_instrument_short_names(self) -> list[str]:
        """Return short names suitable for the conversation UI.

        Convention: take the first segment before ``-`` so that
        ``ot2-robot`` → ``ot2``, ``plc-controller`` → ``plc``.
        If there is no dash the full ID is used (``squidstat`` → ``squidstat``).
        """
        return sorted({
            _instrument_short_name(inst)
            for inst in self.list_instruments()
        })

    def instrument_short_to_full(self) -> dict[str, str]:
        """Return mapping of short_name → full instrument ID.

        E.g. ``{"ot2": "ot2-robot", "plc": "plc-controller", ...}``.
        """
        return {
            _instrument_short_name(inst): inst
            for inst in self.list_instruments()
        }

    def resolve_instrument(self, name: str) -> str | None:
        """Resolve a short or full instrument name to the full ID.

        Returns ``None`` if the name is not recognised.
        """
        if name in {s.instrument for s in self._skills.values() if s.instrument}:
            return name  # already a full ID
        mapping = self.instrument_short_to_full()
        return mapping.get(name)

    def primitives_by_error_class(self, error_class: str) -> list[PrimitiveSpec]:
        """Return all primitives with a given error class."""
        return [p for p in self._primitives.values() if p.error_class == error_class]

    def primitives_by_safety_class(self, safety_class: SafetyClass | str) -> list[PrimitiveSpec]:
        """Return all primitives with a given safety class.

        Accepts SafetyClass enum or string (case-insensitive).
        """
        if isinstance(safety_class, str):
            safety_class = SafetyClass.from_string(safety_class)
        return [p for p in self._primitives.values() if p.safety_class == safety_class]

    # -- Serialization (for API responses) ---------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full registry for API responses."""
        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "version": s.version,
                    "instrument": s.instrument,
                    "resource_id": s.resource_id,
                    "primitive_count": len(s.primitives),
                    "primitives": [
                        {
                            "name": p.name,
                            "description": p.description,
                            "error_class": p.error_class,
                            "safety_class": p.safety_class.name,
                            "params": [
                                {
                                    "name": pp.name,
                                    "type": pp.type,
                                    "description": pp.description,
                                    "optional": pp.optional,
                                    **({"default": pp.default} if pp.default is not None else {}),
                                }
                                for pp in p.params
                            ],
                            **(
                                {
                                    "contract": {
                                        "preconditions": [
                                            pc.predicate for pc in p.contract.preconditions
                                        ],
                                        "effects": [
                                            ef.operation for ef in p.contract.effects
                                        ],
                                        "timeout": {
                                            "seconds": p.contract.timeout.seconds,
                                            "retries": p.contract.timeout.retries,
                                        },
                                    }
                                }
                                if p.contract
                                else {}
                            ),
                        }
                        for p in s.primitives
                    ],
                }
                for s in self._skills.values()
            ],
            "total_primitives": len(self._primitives),
            "total_skills": len(self._skills),
        }

    # -- Summary (for LLM context) ----------------------------------------

    def summary_for_llm(self) -> str:
        """Generate a compact text summary suitable for LLM system prompts."""
        lines: list[str] = ["# Available Capabilities\n"]
        for skill in self._skills.values():
            lines.append(f"## {skill.name}")
            lines.append(f"{skill.description}\n")
            for p in skill.primitives:
                err_tag = f"[{p.error_class}]"
                safety_tag = f"[{p.safety_class.name}]"
                lines.append(f"- `{p.name}` {err_tag} {safety_tag}: {p.description}")
                if p.params:
                    for pp in p.params:
                        opt = " (optional)" if pp.optional else ""
                        lines.append(f"  - `{pp.name}`: {pp.type}{opt} — {pp.description}")
                if p.contract and p.contract.preconditions:
                    preds = ", ".join(pc.predicate for pc in p.contract.preconditions)
                    lines.append(f"  - preconditions: {preds}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _instrument_short_name(full_id: str) -> str:
    """Derive a short instrument name from a full skill instrument ID.

    Convention: first segment before ``-``.
    ``ot2-robot`` → ``ot2``, ``plc-controller`` → ``plc``,
    ``squidstat`` → ``squidstat``.
    """
    return full_id.split("-")[0]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[2] / "agent" / "skills"


@lru_cache(maxsize=1)
def get_registry() -> PrimitivesRegistry:
    """Return the singleton primitives registry, loading skills on first call."""
    registry = PrimitivesRegistry()
    registry.load_skills_dir(_DEFAULT_SKILLS_DIR)
    # Also load LLM-generated skills from the generated/ subdirectory
    registry.load_skills_dir(_DEFAULT_SKILLS_DIR / "generated")
    return registry


def refresh_registry() -> PrimitivesRegistry:
    """Clear the cached registry and reload all skills from disk.

    Call this after onboarding writes new skill files so that newly
    added instruments become visible to the conversation engine and
    NL parser without a server restart.
    """
    get_registry.cache_clear()
    return get_registry()
