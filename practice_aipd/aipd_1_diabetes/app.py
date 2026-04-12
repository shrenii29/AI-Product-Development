import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request
app= Flask(__name__)
scaler= joblib.load('scaler.pkl')
rf= joblib.load('rf.pkl')
dt= joblib.load('dt.pkl')
lr= joblib.load('lr.pkl')
@app.route('/', methods=["GET", "POST"])
def home():
    result=""
    if request.method=="POST":
        Pregnancies= float(request.form['pregnancies'])
        Glucose= float(request.form['glucose'])
        BloodPressure= float(request.form['bp'])
        SkinThickness= float(request.form['st'])
        Insulin= float(request.form['insulin'])
        BMI= float(request.form['bmi']) 
        DiabetesPedigreeFunction= float(request.form['dpf']) 
        Age= float(request.form['age'])

        # Create input array with actual values (not column names)
        x= np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        # Scale the input using the loaded scaler
        x_scaled= scaler.transform(x)
        
        # Make prediction
        pred= rf.predict(x_scaled)[0]

        result= "High risk" if pred==1 else "Normal"

    return render_template("index.html", result=result)
if __name__=="__main__":
    app.run(debug=True)