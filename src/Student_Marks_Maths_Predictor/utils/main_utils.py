import os
import sys

import yaml
import dill

from src.Student_Marks_Maths_Predictor.logger.logger import logging
from src.Student_Marks_Maths_Predictor.exception.expection import CustomException

class MainUtils:
    
    # Read Yaml file and return data as dictionary:
    
    def read_yaml_file(self,filename : str) -> dict:
        logging.info("Entered the read_yaml_file method of the MainUtils class")
        
        try:
            
            if not os.path.exists(filename):
               raise FileNotFoundError(f"The file {filename} doesnot exist")
           
            # If the filename exists then open it..
            with open(filename, "r") as yaml_file:
                return yaml.safe_load(yaml_file)
            
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
    def save_object(self,file_path,obj):
        try:
            
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            with open(file_path,'wb') as file_obj:
                dill.dump(file_path,obj)  
                    
        except Exception as e:
            CustomException(str(e),sys)  
        