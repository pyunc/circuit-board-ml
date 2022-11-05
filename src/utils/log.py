"""Definition of logging lib.

The application log can include debugging messages.
"""
import logging

from utils.config import Config

config = Config()

logging.basicConfig(level=config.log_level)
logger = logging.getLogger()
