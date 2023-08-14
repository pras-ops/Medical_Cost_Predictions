import sys
import pandas as pd
import os
from exception import CustomException
from utils import load_object
from sklearn.decomposition import PCA


from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 age: float,
                 bmi: float,
                 sex: int,
                 children: int,
                 smoker: int,
                 region: int):
         
        self.age = age
        self.bmi = bmi
        self.sex = sex
        self.children = children
        self.smoker = smoker
        self.region = region


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "bmi": [self.bmi],  
                "sex": [self.sex],           
                "children": [self.children],  
                "smoker": [self.smoker],   
                "region": [self.region]    
            }

            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e, sys)
    