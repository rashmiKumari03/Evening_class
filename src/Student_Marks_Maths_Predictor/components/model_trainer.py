import os
import sys

from dataclasses import dataclass
from src.Student_Marks_Maths_Predictor.logger.logger import logging
from src.Student_Marks_Maths_Predictor.exception.expection import CustomException
from src.Student_Marks_Maths_Predictor.utils.main_utils import MainUtils

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)


@dataclass
class Model_Trainer_Config:
    """
    Configuration class to define the path where the trained model will be stored.
    """
    trained_model_file_path = os.path.join("artifacts", "Model_Trainer", "model.pkl")


class Model_Trainer:
    """
    Class responsible for training multiple regression models, evaluating their performance, 
    and selecting the best-performing model based on R-squared score.
    """
    
    def __init__(self):
        logging.info("Initializing Model Trainer...")
        self.model_trainer_config = Model_Trainer_Config()
        self.UTILS = MainUtils()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        """
        Trains multiple regression models, evaluates their performance, 
        selects the best model, and saves it.

        Parameters:
        - train_arr: Training dataset (including target values)
        - test_arr: Testing dataset (including target values)
        - preprocessor_path: Path to the saved preprocessor object

        Returns:
        - R-squared score of the best model
        """
        try:
            logging.info("Splitting the training and test input data...")
            
            # Extracting input features and target variable from train and test sets
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info("Successfully split the data into input features and target values.")
            
            # Defining different regression models for training
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
               
            params = {
                "Linear Regression": {},
                "Random Forest Regressor": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
                "Decision Tree Regressor": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
                "K-Nearest Neighbors Regressor": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                "Gradient Boosting Regressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                "XGBoost Regressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                "CatBoost Regressor": {"iterations": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                "AdaBoost Regressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
            }
            

            logging.info("Starting model evaluation...")

            # Evaluating models using the utility function
            model_report = self.UTILS.evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            logging.info("Model evaluation completed successfully......")
            logging.info(f"model_report is : {model_report}")
            
            # Finding the best-performing model based on R2 score
            best_model_score = max(model_report.values(), key=lambda x: x["Test R2"])["Test R2"] 
            best_model_name = max(model_report, key=lambda k: model_report[k]["Test R2"])
            best_model = models[best_model_name]
           
            
            logging.info(f"Best Model Score: {best_model_score}")
            logging.info(f"Best Model Name: {best_model_name}")
            logging.info(f"The Object Best model: {best_model}")


            logging.info(f"Best model selected: {best_model_name} with R2 Score: {best_model_score:.4f}")

            # Checking if the best model meets a minimum performance threshold
            if best_model_score < 0.5:
                logging.error("No best model found: all models performed below 50%.")
                raise CustomException("No best model found; all models' performance is below 50%.")
            
            logging.info("Best model meets the performance threshold.")

            # Saving the best model for future inference
            logging.info(f"Saving the best model to {self.model_trainer_config.trained_model_file_path}...")
            self.UTILS.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model saved successfully.")

            # Making predictions on test data using the best model
            logging.info("Generating predictions on test data using the best model...")
            predicted = best_model.predict(X_test)

            # Calculating R-squared score for performance evaluation
            r2_sq = r2_score(y_test, predicted)
            logging.info(f"Final R-squared score of the best model on test data: {r2_sq:.4f}")

            return r2_sq

        except Exception as e:
            logging.error(f"Error in Model Training: {str(e)}")
            raise CustomException(str(e), sys)
