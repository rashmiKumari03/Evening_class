import os
import sys

import yaml
import joblib
import dill

from pathlib import Path

from src.Student_Marks_Maths_Predictor.logger.logger import logging
from src.Student_Marks_Maths_Predictor.exception.expection import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

class MainUtils:
    
    @staticmethod
    def read_yaml_file(filename: str) -> dict:
        logging.info("Entered the read_yaml_file method of the MainUtils class")
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"The file {filename} does not exist")
            with open(filename, "r") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
            logging.info(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
        
    @staticmethod
    def save_object(file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, 'wb') as file_obj:
                dill.dump(obj, file_obj) 
        except Exception as e:
            logging.error(f"Error while saving object: {str(e)}")
            raise CustomException(str(e), sys) 
            
    def evaluate_model(self, X_train, y_train, X_test, y_test, models, params=None):
        try:
            logging.info("Starting model evaluation...")
            report = {}
            for model_name, model in models.items():
                logging.info(f"Training and evaluating model: {model_name}")
                if params and model_name in params and params[model_name]:
                    from sklearn.model_selection import GridSearchCV
                    grid_search = GridSearchCV(model, params[model_name], cv=5, scoring="r2", n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

                logging.info(f"{model_name} - Train R2: {train_model_score:.4f}, Test R2: {test_model_score:.4f}, CV Score: {cv_score:.4f}")
                report[model_name] = {"Test R2": test_model_score, "CV Score": cv_score}

            logging.info("Model evaluation completed successfully.")
            return report
        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise CustomException(str(e), sys)
        
    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("Entered the load_object method of MainUtils class")
        try:
            with open(file_path, 'rb') as file_obj:
                obj = dill.load(file_obj)
            logging.info(f"Successfully loaded object of type: {type(obj).__name__}")
            logging.info("Exited the load_object method of MainUtils class")
            return obj
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise CustomException(str(e), sys)