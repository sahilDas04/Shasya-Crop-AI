from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
from flask_cors import CORS
import pandas as pd
import math

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


# Load ideal crop nutrient values
ideal_df = pd.read_csv("crop_nutrient_ideal_values.csv")

# Fertilizer nutrient content & price (per 50 kg bag)
fertilizers = {
    "Urea": {"nutrient_fraction": 0.46, "bag_price": 268},
    "SSP": {"nutrient_fraction": 0.16, "bag_price": 362},
    "MOP": {"nutrient_fraction": 0.60, "bag_price": 900}
}

# Season data (approximate values)
season_conditions = {
    "Kharif": {"temperature": 30, "humidity": 80},
    "Rabi": {"temperature": 20, "humidity": 50},
    "Zaid": {"temperature": 35, "humidity": 60}
}

def recommend_fertilizer(crop, season, land_size, current_N, current_P, current_K):
    # Fetch ideal values for the crop
    row = ideal_df[ideal_df["Crop"].str.lower() == crop.lower()]
    if row.empty:
        return {"error": f"Crop '{crop}' not found in dataset."}
    
    row = row.iloc[0]
    ideal_N, ideal_P, ideal_K = row["N_mean"], row["P_mean"], row["K_mean"]

    # Calculate deficits
    deficits = {
        "N": max(0, ideal_N - current_N),
        "P": max(0, ideal_P - current_P),
        "K": max(0, ideal_K - current_K)
    }

    # If no deficit → no fertilizer needed
    if all(v == 0 for v in deficits.values()):
        return {"Crop": crop, "Message": "No fertilizer needed, soil nutrients are sufficient."}

    # Pick the nutrient with the highest deficit
    major_deficit = max(deficits, key=deficits.get)

    # Map nutrient → fertilizer
    if major_deficit == "N":
        fert_name = "Urea"
        fert_per_ha = deficits["N"] / fertilizers[fert_name]["nutrient_fraction"]
    elif major_deficit == "P":
        fert_name = "SSP"
        fert_per_ha = deficits["P"] / fertilizers[fert_name]["nutrient_fraction"]
    else:
        fert_name = "MOP"
        fert_per_ha = deficits["K"] / fertilizers[fert_name]["nutrient_fraction"]

    # Total fertilizer required for given land size
    total_fert = fert_per_ha * land_size

    # Bags required (50kg per bag, round up)
    bags_needed = math.ceil(total_fert / 50)

    # Total cost
    cost_per_bag = fertilizers[fert_name]["bag_price"]
    total_cost = bags_needed * cost_per_bag

    # Season details
    if season not in season_conditions:
        return {"error": f"Invalid season '{season}'. Choose from Kharif, Rabi, Zaid."}

    season_temp = season_conditions[season]["temperature"]
    season_humidity = season_conditions[season]["humidity"]

    return {
        "Crop": crop,
        "Season": season,
        "Land Size (ha)": land_size,
        "Ideal Temp (°C)": season_temp,
        "Ideal Humidity (%)": season_humidity,
        "Recommended Fertilizer": fert_name,
        "Amount per ha (kg)": round(fert_per_ha, 2),
        "Total Fertilizer (kg)": round(total_fert, 2),
        "Bags Required (50kg)": bags_needed,
        "Estimated Cost (Rs)": total_cost
    }

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



@app.route("/fertilizer-recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        crop = data["crop"]
        season = data["season"]
        land_size = float(data["land_size"])   # New input
        current_N = float(data["N"])
        current_P = float(data["P"])
        current_K = float(data["K"])

        result = recommend_fertilizer(crop, season, land_size, current_N, current_P, current_K)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
