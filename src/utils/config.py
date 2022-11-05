"""Definition of environment variables."""
import os
from typing import Any, List, Optional


class Config:
    """Config class."""

    _instance = None

    def __init__(self):
        """Config constructor."""

        self.env = self.read_env_variable("RR_ENV", "dev", ["dev", "prd"], cast=str)
        self.fallback_feature = self.read_env_variable("RR_FALLBACK_FEATURE", None, None, cast=str)
        self.log_level = self.read_env_variable(
            "RR_LOG_LEVEL", default="INFO", possibilities=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )

    def read_env_variable(
        self, env_name: str, default: Any = None, possibilities: Optional[List[Any]] = None, cast: type = str
    ):
        """Read an environemnt variable.

        Args:
            env_name (str): The name of the environment variable
            default (Any): The default value that the variable will take if it is not set
            possibilities (List): A list with the possibilities allowed. Can be None if it can be anything.
            cast (Type): How to cast the variable. E.g, int, str etc
        """
        var = cast(os.environ.get(env_name, default))
        self.__check_if_value_within_range(env_name, var, possibilities)
        return var

    def __check_if_value_within_range(self, label: str, value: str, range: Optional[List[Any]]):
        """Check if value is within determined range.

        Otherwise raises an error.
        """
        if range is not None and value not in range:
            raise ValueError(f"{label} should be one of the following: {range}. Instead got {value}")
