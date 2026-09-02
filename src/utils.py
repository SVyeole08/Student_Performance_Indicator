import os, sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as fo:
            dill.dump(obj, fo)
    except Exception as e:
        raise CustomException(e, sys)
