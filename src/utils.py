import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(obj, file_path):
    try:
        logging.info(f"Saving data to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Data saved successfully")
        
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            logging.info(f"Training model {print(para)}")

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report

    except Exception as e:
        raise CustomException(e, sys)