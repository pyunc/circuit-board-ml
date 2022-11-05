from src.input.config.base_config import Config


class ModelConfig(Config):
    """Base config object."""

    def __init__(self):

        super().__init__()

        """Class constructor."""

        self.selected_model: str = 'ALTERNATING_LEAST_SQUARE'
