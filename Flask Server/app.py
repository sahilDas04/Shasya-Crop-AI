from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os
import pandas as pd

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "../models")
rf_path = os.path.join(model_dir, "best_randomforest_model.pkl")
gnb_path = os.path.join(model_dir, "best_gaussiannb_model.pkl")

try:
    with open(rf_path, 'rb') as f:
        loaded_rf = pickle.load(f)
except FileNotFoundError:
    loaded_rf = None
    print(f"RandomForest model file not found: {rf_path}")

try:
    with open(gnb_path, 'rb') as f:
        loaded_gnb = pickle.load(f)
except FileNotFoundError:
    loaded_gnb = None
    print(f"GaussianNB model file not found: {gnb_path}")


@app.route('/')
def home():
    return "Crop Recommendation Flask API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    try:
        nitro = float(data.get("N", 0))
        passh = float(data.get("P", 0))
        pota = float(data.get("K", 0))
        temperature = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        ph = float(data.get("ph", 0))
        rain = float(data.get("rain", 0))  
        model_choice = data.get("model", "RandomForestClassifier")
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400

    model = loaded_rf if model_choice == 'RandomForestClassifier' else loaded_gnb
    if not model:
        return jsonify({"error": f"{model_choice} model not loaded."}), 500

    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_df = pd.DataFrame([[nitro, passh, pota, temperature, humidity, ph, rain]], columns=feature_names)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_df)[0]
        top5_idx = np.argsort(proba)[::-1][:5]
        crops = np.array(model.classes_)[top5_idx].tolist()
        return jsonify({"top5_recommended_crops": crops})
    else:
        prediction = model.predict(input_df)[0]
        return jsonify({"recommended_crop": prediction})


if __name__ == '__main__':
    app.run(debug=True)
