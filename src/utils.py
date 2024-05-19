import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import dill


def save_object(obj, file_path):
    try:
        logging.info(f"Saving data to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Data saved successfully")
        
    except Exception as e:
        raise CustomException(e, sys)