"""
UO Template Library.

Contains domain-specific Unit Operation templates for different
experiment types. Templates provide pre-configured UOs with
standard parameters and placeholders.
"""

from typing import Dict, List, Optional

from ..ir import UnitOperation


class TemplateRegistry:
    """
    Registry for UO templates.

    Provides lookup and retrieval of templates by domain and name.
    """
    _instance = None
    _templates: Dict[str, Dict[str, UnitOperation]] = {}

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
    def get(cls, domain: str, name: str) -> Optional[UnitOperation]:
        """Get a template by domain and name. Returns a copy."""
        if domain in cls._templates and name in cls._templates[domain]:
            return cls._templates[domain][name].copy()
        return None

    @classmethod
    def list_domains(cls) -> List[str]:
        """List available domains."""
        return list(cls._templates.keys())

    @classmethod
    def list_templates(cls, domain: str) -> List[str]:
        """List templates in a domain."""
        if domain in cls._templates:
            return list(cls._templates[domain].keys())
        return []

    @classmethod
    def get_all_for_domain(cls, domain: str) -> Dict[str, UnitOperation]:
        """Get all templates for a domain (copies)."""
        if domain in cls._templates:
            return {k: v.copy() for k, v in cls._templates[domain].items()}
        return {}


def get_template(domain: str, name: str) -> Optional[UnitOperation]:
    """Convenience function to get a template."""
    return TemplateRegistry.get(domain, name)


def list_templates(domain: str = None) -> Dict[str, List[str]]:
    """List available templates, optionally filtered by domain."""
    if domain:
        return {domain: TemplateRegistry.list_templates(domain)}
    return {d: TemplateRegistry.list_templates(d) for d in TemplateRegistry.list_domains()}


# Import domain templates to register them
from . import oer

__all__ = [
    "TemplateRegistry",
    "get_template",
    "list_templates",
]
