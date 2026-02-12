"""
Potentiostat Plugin

Supports electrochemistry instruments like SquidStat, Gamry, BioLogic, Autolab.
"""

from .plugin import PotentiostatPlugin
from .parser import PotentiostatParser
from .operations import ElectrochemOperation

__all__ = [
    'PotentiostatPlugin',
    'PotentiostatParser',
    'ElectrochemOperation',
]
