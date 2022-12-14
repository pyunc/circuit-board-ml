"""Generalized Preprocess."""
from __future__ import annotations

import datetime
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from enum import Enum

if TYPE_CHECKING:
    from utils.config import Config


class Encoder(Enum):
    ONE_HOT_ENCODER = "OneHotEncoder"
    ORDINAL_ENCODER = "OrdinalEncoder"


class Preprocess(ABC):
    """Abstract class that does the preprocess of the model."""

    def __init__(
        self,
        config
    ):
        """Definition of Preprocess constructor.

        Args:
            hyperparameters (Hyperparameters): model parameters
        """
        self.config = config

    @abstractmethod
    def run(self):
        """Run abstract method."""
        raise NotImplementedError()