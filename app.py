from flask import Flask,render_template,request,jsonify
import numpy as np
from src.pipeline.data_prediction import PredictionPipeline
import os
import pandas as pd
import joblib


app=Flask(__name__)

@app.route('/',methods=['GET'])
def homepage():
    return render_template('index2.html') 


@app.route('/train',methods=["GET"])
def training():
    os.system("python run_pipeline.py")
    return ("Training Successful")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():

    
    if request.method == 'POST':
            #  reading the inputs given by the user
            Region_code=float(request.form['Region_code'])
            Property_type=float(request.form['Property_type'])
            Bedrooms_str=float(request.form['Bedrooms_str'])
            Year=float(request.form['Year'])
            Month_name=float(request.form['Month_name'])
            Day_name=float(request.form['Day_name'])
            Price_per_bedrooms=float(request.form['Price_per_bedrooms'])

            input_data = [Region_code,Property_type,Bedrooms_str,Year,Month_name,Day_name,Price_per_bedrooms]

            #input_data = np.array(input_data).reshape(1, 7)

            preprocessor = joblib.load("saved_preproccesor/preprocessor.joblib")
            model=joblib.load("saved_model/model.joblib")


            #obj = PredictionPipeline()
            input_data = request.get_json()
            input_data = pd.DataFrame([input_data])
            input_data=preprocessor.transform(input_data)
            prediction = model.predict([input_data])
    
            return render_template("index.html", prediction_text=f"Prediction: {prediction}")


if __name__=='__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)