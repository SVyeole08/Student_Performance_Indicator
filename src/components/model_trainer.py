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

            params = {
                "LinearRegression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False],
                },
                "Ridge": {
                    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "fit_intercept": [True, False],
                },
                "Lasso": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1],
                    "fit_intercept": [True, False],
                    "selection": ["cyclic", "random"],
                    "max_iter": [10000],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto"],
                },
                "SVR": {
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"],
                    "C": [0.1, 1, 10, 100],
                    "epsilon": [0.01, 0.1, 0.2],
                },
                "DecisionTree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20],
                    "max_features": [None, "sqrt", "log2"],
                },
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "criterion": ["squared_error"],
                    "max_depth": [None, 10, 20],
                    "max_features": ["sqrt", 1.0],
                    "bootstrap": [True],
                },
                "AdaBoost": {
                    "learning_rate": [0.01, 0.1, 0.5],
                    "n_estimators": [50, 100, 200],
                    "loss": ["linear", "square"],
                },
                "GradientBoost": {
                    "loss": ["squared_error", "huber"],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200],
                    "subsample": [0.75, 1.0],
                    "max_features": [None, "sqrt"],
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                    "min_child_weight": [1, 3],
                    "subsample": [0.8, 1.0],
                },
                "CatBoost": {
                    "iterations": [100, 300],
                    "learning_rate": [0.05, 0.1],
                    "depth": [4, 6, 8],
                },
            }

            logging.info("Evaluating models....")
            model_report: dict = evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                                models=models, params=params, cv=3)

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
