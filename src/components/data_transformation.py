import sys
import os
from dataclasses import dataclass


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('Artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 
                                  'reading_score']
            
            categorical_columns = ["gender",
                                    "race_ethnicity",
                                    "parental_level_of_education",
                                    "lunch",
                                    "test_preparation_course"]
            
            numerical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ])
            
            categorical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical:{numerical_columns} and Categorical: {categorical_columns} transformers created")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_transformer", numerical_transformer, numerical_columns),
                    ("categorical_transformer", categorical_transformer, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df= pd.read_csv(train_data_path)
            test_df= pd.read_csv(test_data_path)

            logging.info("Read train and test data successfully")
            logging.info("obtaining preprocessor object")

            preprocessing_obj= self.get_data_transformer_object()

            target_column= 'math_score'
            numerical_columns = ['writing_score', 
                                  'reading_score']
            
            input_feature_train_df= train_df.drop(target_column, axis=1)
            target_train_df= train_df[target_column]

            input_feature_test_df= test_df.drop(target_column, axis=1)
            target_test_df= test_df[target_column]

            logging.info("Fitting preprocessor object")

            input_feature_train_df_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_arr= preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_df_arr, target_train_df]
            test_arr= np.c_[input_feature_test_df_arr, target_test_df]

            logging.info("Transformation successful")

            save_object(
                file_path= self.transformation_config.preprocessor_path,
                obj = preprocessing_obj
            )

            return (train_arr,
                     test_arr, 
                     self.transformation_config.preprocessor_path,
                    )

        except Exception as e:
            raise CustomException(e, sys)
        
            
