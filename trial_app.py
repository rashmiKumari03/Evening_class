from src.Student_Marks_Maths_Predictor.logger.logger import logging
from src.Student_Marks_Maths_Predictor.exception.expection import CustomException
import sys

import warnings
warnings.filterwarnings('ignore')

# LOGGER :
# Checking wheather the logger working fine or not?
logging.info("Cross Checking wheather logger is working or not")
logging.info("Hello we are learning Data Science and Machine Learning")

print("Lets Learn ML and this is i am priniting and not logging.")
logging.info("Yeah! My Logger is working welll.")
logging.info("---------------------------------------------------------------------------------------")


# Checking wheather the exception working fine or not?
logging.info("Crosscehcking wheather execption working or not.")

try:
    
    num = 1000/0
    print(num)
    
except Exception as e:
    
    logging.info(CustomException(str(e),sys))
    raise CustomException(str(e),sys)
    
    
    
    
    
    
    
