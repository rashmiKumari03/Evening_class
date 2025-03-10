import os 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO) 

project_name = 'Student_Marks_Maths_Predictor'

list_of_files = [
    
    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    
    # Pipelines
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    
    
    # Utils 
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    
    #Constant
    f"src/{project_name}/constant/__init__.py",
    f"src/{project_name}/constant/main_contant.py",
    
    #Configuration
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/mongo_operation.py",
    
    #Entity
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",
    
    
    #Logger
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/logger/logger.py",
    
    #Exception
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/exception/expection.py",
    
    #yaml files
    "config/schema.yaml",
    "config/model.yaml",
    
    #Some more files.
    "requirements.txt",
    "setup.py",
    "trial_app.py",
    "trial_app_run.py",
    "app.py",
    
    #Web dev 
    "templates/home.html",
    "templates/prediction.html",
    "static"   
]


# Till now these files were in str (string format...) so we have to convert them first into Path .

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir , filename = os.path.split(filepath)
    
    
    # SEE THE REPO....FOLDER...
    # If filedir ==> is Not Empty. ie There is something in the filedir ===> Then make dir with that address/path.
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory : {filedir} for the file name {filename}")
        
    # If the Above path doesnot exist or path size is zero .===> path doesnot exist.
    # Then we have to create empty filepath else filename already exists.
    
    
    # SEE THE FILENAME...
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass
            logging.info(f"Creating empty file : {filepath}")
    else:
            logging.info(f"{filename} already exists")
    
    
    
# Now Execute it using python template.py
