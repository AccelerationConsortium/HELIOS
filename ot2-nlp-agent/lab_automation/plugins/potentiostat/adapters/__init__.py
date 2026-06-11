"""
Potentiostat Hardware Adapters

Vendor-specific adapters are optional: lab-internal integrations may be
kept out of the public distribution. The plugin degrades gracefully when
no adapter package is present.
"""

try:
    from .squidstat import SquidStatAdapter  # noqa: F401
    __all__ = ['SquidStatAdapter']
except ImportError:  # pragma: no cover - vendor adapter not installed
    SquidStatAdapter = None
    __all__ = []
