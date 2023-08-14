import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "Medical_Cost_Predictions"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/utils/__init__.py",
    f"src/pipeline/__init__.py",
    f"src/logger.py",
    f"src/exception.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "Dockerfile"
    "requirements.txt",
    "setup.py",
    "Notebook/trial.ipynb",
    "templates/index.html"
]
''' "f"formatted string in Python using an f-string. It is used 
    to dynamically create a string by substituting the value 
    of the project_name variable into the string. '''

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename =os.path.split(filepath)
# Path is used to work in windows

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")
## It checks wether any code or not
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")  