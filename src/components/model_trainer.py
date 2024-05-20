import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (RandomForestRegressor
                                , GradientBoostingRegressor
                                , AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("Artifacts", "trained_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1], 
                test_array[:,:-1], test_array[:,-1]
            )
            models = {
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighbors": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "XGB": XGBRegressor(),
                "CatBoost": CatBoostRegressor()
            }
        
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)


            best_model_score=max(sorted(model_report.values()))

            best_model_name= list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No model performed well", sys)
            logging.info(f"Best model is {best_model_name} with score {best_model_score}")

            save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
            )

            predicted= best_model.predict(X_test)

            r2= r2_score(y_test, predicted)
            return r2


        except Exception as e:
            raise CustomException(e, sys)



