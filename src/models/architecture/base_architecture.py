from abc import ABC, abstractmethod

class BaseArchitecture(ABC):
    """Base Data Source class."""

    def __init__(
        self,
        config: str
    ):
        """Class constructor."""
        self.config = config

    @abstractmethod
    def build(self):
        """Build abstract method."""
        raise NotImplementedError()
