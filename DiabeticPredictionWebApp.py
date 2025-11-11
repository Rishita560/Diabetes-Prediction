# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 12:52:12 2025

@author: sahur
"""

import numpy as np
import pickle 
import streamlit as st

#loading the saved model
loded_model = pickle.load(open("diabetic_trained_model.sav", 'rb'))

#creating a funnction fro prediction
def diabetic_prediction(input_data):
    
    #changing input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "Person is not diabetic"
    else:
        return "Person is diabetic"
    
    
def main():
    
    #giving title
    st.title("Diabetes Prediction Web App")
    
    #getting input data from user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the person")
    
    #code for prediction
    diagnosis = ''
    
    #creating button for prediction
    if st.button("Diabetes test Prediction"):
        diagnosis = diabetic_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    

    
