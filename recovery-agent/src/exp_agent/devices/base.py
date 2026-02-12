from abc import ABC, abstractmethod
from ..core.types import DeviceState, Action

class Device(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def read_state(self) -> DeviceState: ...

    @abstractmethod
    def execute(self, action: Action) -> None: ...

    @abstractmethod
    def health(self) -> bool: ...
