# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 12:52:12 2025

@author: sahur
"""

import numpy as np
import pickle 
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open("diabetic_trained_model.sav", 'rb'))

#creating a funnction fro prediction
def diabetic_prediction(input_data):
    
    #changing input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # prediction (0/1)
    prediction = loaded_model.predict(input_data_reshaped)

    #probability prediction (confidence)
    probability = loaded_model.predict_proba(input_data_reshaped)[0][1] #prob of diabetic class
    
    return prediction[0], probability
    
    
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

        if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.warning("Please fill all fields before prediction.")
            return

        try:
            # Convert all inputs from string to float/int before passing to prediction
            input_list = [
                float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), 
                float(BMI), float(DiabetesPedigreeFunction), float(Age)
            ]
            diagnosis, proba = diabetic_prediction(input_list)
            
            confidence = round(proba * 100, 2)
            proba_percent = int(round(proba * 100)) #integer 0-100 for progress bar

            #risk level scale
            if proba < 0.30:
                risk_level = "Low Risk"
                color = "green"
            elif proba < 0.70:
                risk_level = "Moderate Risk"
                color = "orange"
            else:
                risk_level = "High Risk"
                color = "red"

            #shows final diagnosis
            if diagnosis == 1:
                st.error("Person is Diabetic")
            else:
                st.success("Person is Not Diabetic")

            #shows confidence level
            st.markdown(
                f"<h5> Confidence Score: {confidence}% </h5>",
                unsafe_allow_html=True
            )
            
            st.progress(proba_percent)            
            
            #shows risk level
            st.markdown(
                f"<h4> Risk Level: <span style='color:{color}; font-weight:bold;'>{risk_level}</span></h4>",
                unsafe_allow_html=True
            )
            
        except ValueError:
            #case where a user enters non-numeric text
            st.error("**Input Error:** Please ensure all fields are filled with numeric values.")

if __name__ == '__main__':
    main()

