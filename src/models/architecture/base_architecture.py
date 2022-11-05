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
        function_config: str,
        model_config: str
    ):
        """Class constructor."""
        self.function_config = function_config
        self.model_config = model_config

    @abstractmethod
    def run(self):
        """Run abstract method."""
        raise NotImplementedError()
