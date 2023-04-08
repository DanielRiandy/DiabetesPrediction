import streamlit as st 
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd 
import numpy as np
import pickle
from datetime import datetime

st.title("Diabetes Classifier 🔬")

# st.sidebar.write(f'Last edited on {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
# add_selectbox = st.sidebar.selectbox(
#     'What do you want to do?',
#     ('Summary', 'Prediction')
# )

model_1 = pickle.load(open(r'./Artifacts/model-1.bin', "rb"))
model_2 = pickle.load(open(r'./Artifacts/model-2.bin', "rb"))
scaler_1 = pickle.load(open(r'./Artifacts/scaler.bin', "rb"))
scaler_2 = pickle.load(open(r'./Artifacts/scaler-2.bin', "rb"))


# if add_selectbox == 'Prediction':
col1, col2 = st.columns(2)

pregnancies_ = col1.number_input(
    "Pregnancies (Number of times pregnant)", min_value = 0, step= 1
)

glucose_ = col1.number_input(
    "Glucose concentration", min_value = 0, step= 1
)

bp_ = col1.number_input(
    "Diastolic blood pressure (mm Hg)", min_value = 0, step= 1
)

st_ = col1.number_input(
    "Triceps skin fold thickness (mm)", min_value = 0, step= 1
)

ins_ = col2.number_input(
    "2-Hour serum insulin (mu U/ml)", min_value = 0, step= 1
)

bmi_ = col2.number_input(
    "Body mass index", min_value = 0.0, step= .1
)

age_ = col2.number_input(
    "Your Age (years)", min_value = 0, step= 1
)

dpf_ = col2.number_input(
    "Diabetes pedigree function", min_value = 0.0, 
    step= .001, value= .001, format="%f"
)

unseen_ = np.array([pregnancies_, glucose_, bp_, st_,
                    ins_, bmi_, float(dpf_), age_]).reshape(1,-1)

unseen_final = scaler_2.transform(unseen_)


result = "None"
if st.button(
    "Get Result"
):
    result = model_1.predict(unseen_final)[0][0]

if result != "None":
    if result > .5:
        pred = 'Diabete Detected!'
        score = result
    else :
        pred = 'Clean'
        score = 1 - result
    
    if 0.0 in unseen_:
        st.warning('Features invalid! Please fill the features with non-zero values', icon="⚠️")
    else :
        st.success(f"{pred} | Confidence Level: {round((score*100),3)}%", icon="✅")