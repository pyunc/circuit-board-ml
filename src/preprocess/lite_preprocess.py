"""Random Forest Regressor."""
from src.preprocess.preprocess import Preprocess
from sklearn.ensemble import RandomForestRegressor
from utils.log import logger


class LitePreprocess(Preprocess):
    """Preprocess class."""

    def __init__(self, hyperparameters):
        """Definition of Preprocess constructor.

        Args:
            
        """
        super().__init__(hyperparameters)

        pass
    