import sys
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.Student_Marks_Maths_Predictor.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.Student_Marks_Maths_Predictor.logger.logger import logging
from src.Student_Marks_Maths_Predictor.exception.expection import CustomException
from src.Student_Marks_Maths_Predictor.components.data_ingestion import Data_Ingestion
from src.Student_Marks_Maths_Predictor.components.data_transformation import Data_Transformation
from src.Student_Marks_Maths_Predictor.components.model_trainer import Model_Trainer

import warnings
warnings.filterwarnings("ignore")


# Entry Point
application = Flask(__name__)
app = application

# Home Route
@app.route('/')
def home():
    return render_template("home.html")

# Prediction Route
@app.route('/prediction', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('prediction.html')
    else:
        try:
            # Capture form data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            # Convert to DataFrame
            df = data.get_data_as_frame()
            logging.info(f"Received Data: \n{df}")

            # Make Prediction
            prediction_pipeline = PredictPipeline()
            predicted_math_score = prediction_pipeline.predict(df)
            result = np.round(predicted_math_score[0],2)


            logging.info(f"Predicted Math Score: {result}")

            return render_template('prediction.html', result=result)

        except Exception as e:
            logging.exception("Error occurred while predicting")
            return f"Error in prediction: {str(e)}"



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

        logging.info("Starting Flask App...")
        app.run(port=8080, debug=False)

    except Exception as e:
        logging.error("An error occurred in the main pipeline.")
        raise CustomException(str(e), sys)