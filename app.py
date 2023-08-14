from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from exception import CustomException
from utils import load_object


application = Flask(__name__)


app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')    

    else:
        
        data = CustomData(
            age=float(request.form.get('age')),
            bmi=float(request.form.get('bmi')),
            sex=int(request.form.get('sex')),
            children=int(request.form.get('children')),
            smoker=int(request.form.get('smoker')),
            region=int(request.form.get('region'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("Results:", results)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)