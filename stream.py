import numpy as np
import joblib
import streamlit as st

scalar = joblib.load("scaled.pkl")
model = joblib.load("model_1_scaled.pkl")

# STREAMLIT TITLE
st.title("Machine Learning Model")
# writing a promt
st.write("Enter your Medical Details")

#  dEFINE THE INPUT FIELDS
st.sidebar.header("Your Medical Records")

# Input from user
preg = st.sidebar.number_input(
    "preg", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)
plas = st.sidebar.number_input(
    "plas", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)
pres = st.sidebar.number_input(
    "pres", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)
skin = st.sidebar.number_input(
    "skin", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)
test = st.sidebar.number_input(
    "test", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)
mass = st.sidebar.number_input(
    "mass", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)
pedi = st.sidebar.number_input(
    "pedi", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)
age = st.sidebar.number_input(
    "age", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)

input_array = np.array([[preg, plas, pres, skin, test, mass, pedi, age]])

scaled_input = scalar.transform(input_array)

if st.sidebar.button("predict"):
    prediction = model.predict(
        scaled_input
    )  # do the prediction once predict button is pressed
    st.success(f"Prediction :{prediction[0]}")  # display output
