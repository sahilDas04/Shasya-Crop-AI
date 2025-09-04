
import streamlit as st
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "../models")
rf_path = os.path.join(model_dir, "best_randomforest_model.pkl")
gnb_path = os.path.join(model_dir, "best_gaussiannb_model.pkl")

try:
    with open(rf_path, 'rb') as f:
        loaded_rf = pickle.load(f)
except FileNotFoundError:
    loaded_rf = None
    st.error(f"RandomForest model file not found: {rf_path}")

try:
    with open(gnb_path, 'rb') as f:
        loaded_gnb = pickle.load(f)
except FileNotFoundError:
    loaded_gnb = None
    st.error(f"GaussianNB model file not found: {gnb_path}")

st.title('Crop Recommendation')

nitro = st.number_input('Nitrogen in soil')
passh = st.number_input('Phosphorus in soil')
pota = st.number_input('Potassium in soil')
temperature = st.number_input('Temperature')
humidity = st.number_input('Humidity')
ph = st.number_input('Ph')
rain = st.number_input('rain Fall')

model_choice = st.selectbox('Select Model', ['RandomForestClassifier', 'GaussianNB'])

if st.button('Predict'):
    data = np.array([nitro, passh, pota, temperature, humidity, ph, rain]).reshape(1, -1)
    if model_choice == 'RandomForestClassifier':
        if loaded_rf:
            predict = loaded_rf.predict(data)
            st.success(f"Recommended crop for this season: {predict[0]}")
        else:
            st.error("RandomForest model not loaded.")
    elif model_choice == 'GaussianNB':
        if loaded_gnb:
            predict = loaded_gnb.predict(data)
            st.success(f"Recommended crop for this season: {predict[0]}")
        else:
            st.error("GaussianNB model not loaded.")
