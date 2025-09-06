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
    import pandas as pd
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    data = np.array([nitro, passh, pota, temperature, humidity, ph, rain]).reshape(1, -1)
    model = loaded_rf if model_choice == 'RandomForestClassifier' else loaded_gnb
    if model:
        if hasattr(model, 'predict_proba'):
            data = pd.DataFrame([[nitro, passh, pota, temperature, humidity, ph, rain]], columns=feature_names)
            proba = model.predict_proba(data)[0]
            top5_idx = np.argsort(proba)[::-1][:5]
            crops = np.array(model.classes_)[top5_idx]
            st.success(f"Top 5 recommended crops: {', '.join(crops)}")
        else:
            predict = model.predict(data)
            st.success(f"Recommended crop for this season: {predict[0]}")
    else:
        st.error(f"{model_choice} model not loaded.")
