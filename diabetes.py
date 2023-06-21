# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st

#loading trained saved model

loaded_model=pickle.load(open('C:/Users/Suresh/trained_model.sav','rb'))

#creating prediction
def diabetics_pred(input_data):
    #converting input data to numerical values
    input_data=[float(x) for x in input_data]
    
    #changing input data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)
    
    #reshaping the array as we are predicting only for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    
    prediction=loaded_model.predict(input_data_reshaped)
    
    if prediction[0]==0:
        return 'The user is a non diabetic'
    else:
        return 'The user is a diabetic'
    
    
def main():
    #giving title
    st.title("Diabetic prediction")
    
    #getting input data from the user
    Pregnancies=st.text_input('pregnancies')
    Glucose=st.text_input('Glucose')
    BloodPressure=st.text_input('BloodPressure')
    SkinThickness=st.text_input('SkinThickness')
    Insulin=st.text_input('Insulin')
    BMI=st.text_input('BMI')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction')
    Age=st.text_input('Age')
    
    #code for prediction
    result=""
    
    if st.button("user result"):
        input_data=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
        result=diabetics_pred(input_data)
        
    st.success(result)
    
    
if __name__=='__main__':
    main()
    