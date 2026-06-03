"""
Potentiostat Plugin

Supports electrochemistry instruments like SquidStat, Gamry, BioLogic, Autolab.
"""

from .operations import ElectrochemOperation
from .parser import PotentiostatParser
from .plugin import PotentiostatPlugin

__all__ = [
    'PotentiostatPlugin',
    'PotentiostatParser',
    'ElectrochemOperation',
]
