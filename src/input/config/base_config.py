from __future__ import annotations

import datetime
import os,ast
from abc import ABC, abstractmethod
from utils.logger import get_logger

class Config(object):
    _instance = None

    def __init__(self):

        """Class constructor."""

        self.bucket_s3: str = os.environ.get('BUCKET_S3', 'dev-ml-sagemaker')

        # project
        self.project_s3: str = os.environ.get('PROJECT_S3', 'quality-platform')

        # country
        self.country: str = os.environ.get('COUNTRY', 'USA')

        # resolution h3
        self.resolution: int = ast.literal_eval(os.environ.get('RESOLUTION', '7'))

        self.downloaded_data_path: str = '/home/paulo/Documents/project-practical-mlops/circuit-board-ml/src/input/data/downloaded_data'

        self.organized_data_path: str = '/home/paulo/Documents/project-practical-mlops/circuit-board-ml/src/input/data/organized_data'

        self.model_save_path: str = '/home/paulo/Documents/project-practical-mlops/circuit-board-ml/src/input/data/saved_model'

        self.preprocess: str = 'LITEPREPROCESS'

        self.selected_model: str = 'MOBILENET'
                                            
        self.logger = get_logger()
        

        