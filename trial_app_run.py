import sys
from src.Student_Marks_Maths_Predictor.logger.logger import logging
from src.Student_Marks_Maths_Predictor.exception.expection import CustomException
from src.Student_Marks_Maths_Predictor.components.data_ingestion import Data_Ingestion
from src.Student_Marks_Maths_Predictor.components.data_transformation import Data_Transformation
from src.Student_Marks_Maths_Predictor.components.model_trainer import Model_Trainer

if __name__ == "__main__":
    try:
        logging.info("Starting the Machine Learning Pipeline...")

        # Step 1: Data Ingestion
        logging.info("Starting the Data Ingestion process...")
        data_ingestion = Data_Ingestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed successfully. Train Path: {train_path}, Test Path: {test_path}")

        # Step 2: Data Transformation
        logging.info("Starting the Data Transformation process...")
        data_transformation = Data_Transformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
        logging.info(f"Data Transformation completed successfully. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training
        logging.info("Starting the Model Training process...")
        model_trainer = Model_Trainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
        logging.info(f"Model Training completed successfully. Best Model R2 Score: {r2_score:.4f}")

        logging.info("Machine Learning Pipeline execution completed successfully!")

    except Exception as e:
        logging.error("An error occurred in the main pipeline.")
        raise CustomException(str(e), sys)
