import sys
import os
from dataclasses import dataclass
from src.Student_Marks_Maths_Predictor.logger.logger import logging
from src.Student_Marks_Maths_Predictor.exception.expection import CustomException
from src.Student_Marks_Maths_Predictor.utils.main_utils import MainUtils
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class Data_Transformation_Config:
    """
    Configuration class for data transformation process.
    Stores the file path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path: str = os.path.join("artifacts", "Data_Transformation", "preprocessor.pkl")


class Data_Transformation:
    def __init__(self):
        """
        Initializes the Data Transformation class by setting up configuration and utility objects.
        Reads the schema file containing column details.
        """
        logging.info("Initializing Data Transformation class...")
        self.data_transformation_config = Data_Transformation_Config()
        self.UTILS = MainUtils()
        self.SCHEMA_FILE = self.UTILS.read_yaml_file(filename="config/schema.yaml")
        logging.info("Schema file loaded successfully.")

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessor object for data transformation.
        Includes separate pipelines for numerical and categorical features.
        """
        try:
            logging.info("Fetching numerical and categorical feature names from schema file...")
            
            numerical_cols = self.SCHEMA_FILE['numerical_features']
            categorical_cols = self.SCHEMA_FILE['categorical_features']
            logging.info(f"Numerical Features Identified: {numerical_cols}")
            logging.info(f"Categorical Features Identified: {categorical_cols}")

            # Defining pipeline for numerical features
            logging.info("Creating pipeline for numerical feature transformation...")
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info("Numerical feature pipeline created successfully.")

            # Defining pipeline for categorical features
            logging.info("Creating pipeline for categorical feature transformation...")
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical feature pipeline created successfully.")

            # Combining numerical and categorical pipelines
            logging.info("Combining numerical and categorical pipelines...")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_cols),
                    ('categorical_pipeline', categorical_pipeline, categorical_cols)
                ]
            )
            logging.info("Data transformation pipeline successfully created.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error in creating data transformer object: {str(e)}")
            raise CustomException(str(e), sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test datasets, applies transformations, and saves the preprocessor object.
        Returns transformed training and testing arrays along with the preprocessor file path.
        """
        try:
            logging.info("Reading training and testing datasets...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")

            logging.info("Initializing data transformation process...")
            preprocessor_obj = self.get_data_transformer_object()

            # Extract target column name from schema file
            target_column_name = self.SCHEMA_FILE['target_feature']
            logging.info(f"Target column identified: {target_column_name}")

            # Separating input and target features for training data
            logging.info("Separating input and target features for training data...")
            input_feature_train_df = train_df.drop(columns=target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            logging.info("Training data successfully separated into input and target features.")

            # Separating input and target features for testing data
            logging.info("Separating input and target features for testing data...")
            input_feature_test_df = test_df.drop(columns=target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Testing data successfully separated into input and target features.")

            # Applying transformations to train and test data
            logging.info("Applying transformations to training and testing data...")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info("Transformation applied successfully.")

            # Merging input features with target feature for both train and test datasets
            logging.info("Merging transformed input features with target feature...")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Data successfully merged into final arrays.")

            # Saving the preprocessor object for future inference
            logging.info("Saving the preprocessor object...")
            self.UTILS.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(str(e), sys)