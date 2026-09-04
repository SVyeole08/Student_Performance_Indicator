import os, sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components import data_transformation
from src.components import model_trainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("Entered the data ingestion method.")

        try:
            df = pd.read_csv("src/notebooks/data/student_data.csv")
            logging.info("Read data as Dataframe")

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)

            df.to_csv(self.ingestionConfig.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated.")
            train_set, test_set = train_test_split(df, test_size=.2, random_state=42)

            train_set.to_csv(self.ingestionConfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestionConfig.test_data_path, index=False, header=True)
            logging.info("Data ingested successfully.")

            return (
                self.ingestionConfig.train_data_path, self.ingestionConfig.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.init_data_ingestion()

    data_transform = data_transformation.DataTransformation()
    train_arr, test_arr, _ = data_transform.init_data_transformation(train_data, test_data)

    model_train = model_trainer.ModelTrainer()
    print(model_train.init_model_training(train_arr=train_arr, test_arr=test_arr))
