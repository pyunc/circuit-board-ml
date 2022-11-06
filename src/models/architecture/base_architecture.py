import os
import tempfile
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

import pandas as pd


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
