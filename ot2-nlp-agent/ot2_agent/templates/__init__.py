"""
UO Template Library.

Contains domain-specific Unit Operation templates for different
experiment types. Templates provide pre-configured UOs with
standard parameters and placeholders.
"""

from ..ir import UnitOperation


class TemplateRegistry:
    """
    Registry for UO templates.

    Provides lookup and retrieval of templates by domain and name.
    """
    _instance = None
    _templates: dict[str, dict[str, UnitOperation]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._templates = {}
        return cls._instance

    @classmethod
    def register(cls, domain: str, name: str, template: UnitOperation):
        """Register a template."""
        if domain not in cls._templates:
            cls._templates[domain] = {}
        cls._templates[domain][name] = template

    @classmethod
    def get(cls, domain: str, name: str) -> UnitOperation | None:
        """Get a template by domain and name. Returns a copy."""
        if domain in cls._templates and name in cls._templates[domain]:
            return cls._templates[domain][name].copy()
        return None

    @classmethod
    def list_domains(cls) -> list[str]:
        """List available domains."""
        return list(cls._templates.keys())

    @classmethod
    def list_templates(cls, domain: str) -> list[str]:
        """List templates in a domain."""
        if domain in cls._templates:
            return list(cls._templates[domain].keys())
        return []

    @classmethod
    def get_all_for_domain(cls, domain: str) -> dict[str, UnitOperation]:
        """Get all templates for a domain (copies)."""
        if domain in cls._templates:
            return {k: v.copy() for k, v in cls._templates[domain].items()}
        return {}


def get_template(domain: str, name: str) -> UnitOperation | None:
    """Convenience function to get a template."""
    return TemplateRegistry.get(domain, name)


def list_templates(domain: str = None) -> dict[str, list[str]]:
    """List available templates, optionally filtered by domain."""
    if domain:
        return {domain: TemplateRegistry.list_templates(domain)}
    return {d: TemplateRegistry.list_templates(d) for d in TemplateRegistry.list_domains()}


# Import domain templates to register them. Domain template packages are
# optional: lab-specific protocol templates may be kept out of the public
# distribution, in which case the registry simply starts empty for them.
try:
    from . import oer  # noqa: E402, F401
except ImportError:  # pragma: no cover - template pack not installed
    oer = None

__all__ = [
    "TemplateRegistry",
    "get_template",
    "list_templates",
]
