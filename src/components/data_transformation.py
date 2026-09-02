import os, sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    pre_processor_object_path: str = os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.dataTransformationConfig = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                                    'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('oneHotEncoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Numerical columns: {numeric_features}')
            logging.info(f'Categorical columns: {categorical_features}.')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def init_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data.")

            logging.info("Obtaining preprocessor object...")
            preprocessor_obj = self.get_data_transformer_object()
            target = 'math_score'
            num_cols = ['writing_score', 'reading_score']
            if_train_df = train_df.drop(target, axis=1)
            if_test_df = test_df.drop(target, axis=1)
            tf_train_df = train_df[target]
            tf_test_df = test_df[target]

            logging.info('Applying preprocessing object on train and test dataframe......')
            if_train_arr = preprocessor_obj.fit_transform(if_train_df)
            if_test_arr = preprocessor_obj.transform(if_test_df)

            train_arr = np.c_[
                if_train_arr, np.array(tf_train_df)
            ]
            test_arr = np.c_[
                if_test_arr, np.array(tf_test_df)
            ]
            logging.info('Saved preprocessing object.')

            save_obj(
                file_path=self.dataTransformationConfig.pre_processor_object_path,
                obj=preprocessor_obj
            )

            return (
                train_arr, test_arr, self.dataTransformationConfig.pre_processor_object_path
            )
        except Exception as e:
            raise CustomException(e, sys)
