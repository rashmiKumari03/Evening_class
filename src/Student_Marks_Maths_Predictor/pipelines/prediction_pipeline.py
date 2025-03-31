import sys
import pandas as pd
from src.Student_Marks_Maths_Predictor.exception.expection import CustomException
from src.Student_Marks_Maths_Predictor.logger.logger import logging
from src.Student_Marks_Maths_Predictor.utils.main_utils import MainUtils


utils = MainUtils()

class PredictPipeline:
    def __init__(self):
        pass
       

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading preprocessor and model...")
            
            preprocessor_path = "artifacts\Data_Transformation\preprocessor.pkl"
            model_path = "artifacts\Model_Trainer\model.pkl"

            # Load the preprocessor and model
            preprocessor = utils.load_object(preprocessor_path)
            model = utils.load_object(model_path)
            
            logging.info(f"preprocessor is : {preprocessor}")
            logging.info(f"Model is : {model}")
    

            logging.info("Preprocessing input features...")
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions...")
            predictions = model.predict(data_scaled)

            logging.info("Prediction completed successfully.")
            return predictions

        except Exception as e:
            logging.exception("Error during prediction")
            raise CustomException(str(e), sys)


class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str,
                test_preparation_course: str, reading_score: int, writing_score: int):
        
        """Initialize custom data with user-provided values."""
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self) -> pd.DataFrame:
        """Convert user input into a pandas DataFrame."""
        try:
            custom_data_dict = {
                "gender": self.gender,
                "race_ethnicity": self.race_ethnicity,
                "parental_level_of_education": self.parental_level_of_education,
                "lunch": self.lunch,
                "test_preparation_course": self.test_preparation_course,
                "reading_score": self.reading_score,
                "writing_score": self.writing_score
            }

            return pd.DataFrame([custom_data_dict])  # Wrap in a list to create a proper DataFrame

        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e), sys)
