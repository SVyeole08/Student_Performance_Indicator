import os, sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_model


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifact', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.modelTrainerConfig = ModelTrainerConfig()

    def init_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test data.")
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "GradientBoost": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
            }

            logging.info("Evaluating models....")
            model_report: dict = evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                                models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info("Taken best model.")

            if best_model_score < 0.60:
                raise CustomException("No best model found.")
            logging.info("Found best model on both training and testing dataset.")

            save_obj(self.modelTrainerConfig.train_model_file_path, obj=best_model)
            logging.info("Saved the model.")

            prediction = best_model.predict(X_test)
            score_r2 = r2_score(y_test, prediction)

            return score_r2
        except Exception as e:
            raise CustomException(e, sys)
