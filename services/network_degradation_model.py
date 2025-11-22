# services/network_degradation_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Simulated historical training data
def get_training_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "latency": np.random.normal(20, 5, 200),
        "packet_loss": np.random.normal(0.5, 0.2, 200),
        "jitter": np.random.normal(5, 2, 200),
        "throughput": np.random.normal(100, 20, 200),
        "label": np.random.choice([0, 1], 200, p=[0.85, 0.15])  # 1 = degradation
    })
    return data

def train_model():
    data = get_training_data()
    X = data[["latency", "packet_loss", "jitter", "throughput"]]
    y = data["label"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Save both model and scaler for reuse
    joblib.dump(model, "network_degradation_model.pkl")
    joblib.dump(scaler, "network_scaler.pkl")

def predict_degradation(input_data: dict):
    model = joblib.load("network_degradation_model.pkl")
    scaler = joblib.load("network_scaler.pkl")

    df = pd.DataFrame([input_data])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "degradation_predicted": bool(prediction),
        "confidence": round(probability * 100, 2)
    }

# Only run this once to train the model initially
if __name__ == "__main__":
    train_model()
