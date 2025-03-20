from src.Student_Marks_Maths_Predictor.logger.logger import logging
from src.Student_Marks_Maths_Predictor.exception.expection import CustomException
from src.Student_Marks_Maths_Predictor.components.data_ingestion import Data_Ingestion
from src.Student_Marks_Maths_Predictor.components.data_transformation import Data_Transformation
import sys

if __name__ == "__main__":
    try:
        logging.info("Starting the data ingestion process...")
        data_ingestion = Data_Ingestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed successfully. Train Path: {train_path}, Test Path: {test_path}")
        
        logging.info("Starting the data transformation process...")
        data_transformation = Data_Transformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
        logging.info(f"Data transformation completed successfully. Preprocessor saved at: {preprocessor_path}")
        
    except Exception as e:
        logging.error("An error occurred in the main pipeline.")
        raise CustomException(str(e), sys)