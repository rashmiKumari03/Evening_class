import os
import sys

from src.Student_Marks_Maths_Predictor.exception.expection import CustomException
from src.Student_Marks_Maths_Predictor.logger.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # Used to create Class Variables.


# Any inputs required in Data Ingestion will be provided through this Data_Ingestion_Config

@dataclass
class Data_Ingestion_Config : 
    train_data_path: str = os.path.join('artifacts','Data_Ingestion','train.csv')
    test_data_path : str = os.path.join('artifacts','Data_Ingestion','test.csv')
    raw_data_path  : str = os.path.join('artifacts','Data_Ingestion','raw.csv')
    
    
# Main Class ie. DataIngestion

class Data_Ingestion:
    
    def __init__(self):
        
        self.data_ingestion_config = Data_Ingestion_Config()  #This consists of 3 values..
        
    def initiate_data_ingestion(self):
        logging.info("Entered the initiate_data_ingestion method of Data_Ingestion Class")
        try:
            df = pd.read_csv(r"dataset/raw.csv")
            logging.info('Read the dataset as the DataFrame.')
            
            # Lets create the Paths...
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train Test Split Initiated.!!!")
            
            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data completed...")
            
            return (self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path)
        
        except Exception as e:
            logging.error(CustomException(str(e),sys))
            CustomException(str(e),sys)
        



