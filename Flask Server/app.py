from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "../models")
rf_path = os.path.join(model_dir, "best_randomforest_model.pkl")

loaded_rf = None
try:
    with open(rf_path, 'rb') as f:
        loaded_rf = pickle.load(f)
except FileNotFoundError:
    print(f"RandomForest model file not found: {rf_path}")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Recommendation Flask API is running with RandomForest!"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        nitro = float(data.get("N", 0))
        passh = float(data.get("P", 0))
        pota = float(data.get("K", 0))
        temperature = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        ph = float(data.get("ph", 0))
        rain = float(data.get("rain", 0))
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400

    if not loaded_rf:
        return jsonify({"error": "RandomForest model not loaded."}), 500

    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_df = pd.DataFrame([[nitro, passh, pota, temperature, humidity, ph, rain]], columns=feature_names)

    if hasattr(loaded_rf, 'predict_proba'):
        proba = loaded_rf.predict_proba(input_df)[0]
        top5_idx = np.argsort(proba)[::-1][:5]
        crops = np.array(loaded_rf.classes_)[top5_idx].tolist()
        return jsonify({"top5_recommended_crops": crops})
    else:
        prediction = loaded_rf.predict(input_df)[0]
        return jsonify({"recommended_crop": prediction})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
