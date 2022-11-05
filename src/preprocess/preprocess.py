"""Generalized Models."""
from __future__ import annotations

import datetime
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from utils.enums import ExtendedEnum

if TYPE_CHECKING:
    from utils.config import Config
    from utils.hyperparameters import Hyperparameters  # type: ignore


class Encoder(ExtendedEnum):
    ONE_HOT_ENCODER = "OneHotEncoder"
    ORDINAL_ENCODER = "OrdinalEncoder"


class Preprocess(ABC):
    """Abstract class that does the learning process of the model."""

    def __init__(self, hyperparameters: Hyperparameters):
        """Definition of Ranking Model constructor.

        Args:
            hyperparameters (Hyperparameters): model parameters
        """
        pass