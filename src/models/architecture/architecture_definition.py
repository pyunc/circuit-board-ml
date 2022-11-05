"""This abstract factory is used to assign datasource prodcuts."""
from enum import Enum

from src.models.architecture.mobile_net import MobileNet


class ArchsList(Enum):
    """Class used to assign models."""

    MOBILENET = MobileNet

    
