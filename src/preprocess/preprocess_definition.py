"""This script is used to assign preprocess pipelines."""

from utils.enums import ExtendedEnum

from src.preprocess.lite_preprocess import LitePreprocess

class RankingModels(ExtendedEnum):

    """Class used to assign preprocesser."""

    LITEPREPROCESS = LitePreprocess

