"""This script is used to assign preprocess pipelines."""

from utils.enums import ExtendedEnum

from preprocess.lite_preprocess import LitePreprocess

class PreprocessingList(ExtendedEnum):

    """Class used to assign preprocesser."""

    LITEPREPROCESS = LitePreprocess

