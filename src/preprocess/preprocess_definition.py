"""This script is used to assign preprocess pipelines."""

from enum import Enum

from preprocess.lite_preprocess import LitePreprocess

class PreprocessingList(Enum):

    """Class used to assign preprocesser."""

    LITEPREPROCESS = LitePreprocess

