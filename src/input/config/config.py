from __future__ import annotations

import datetime
import os
from abc import ABC, abstractmethod

class Config(object):
    _instance = None

    def __init__(self):
        # self.model_path = os.environ.get('modelPath', '/opt/ml/model')

        pass