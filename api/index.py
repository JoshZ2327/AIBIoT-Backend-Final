from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import random
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet  # New: Facebook Prophet for better forecasting
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import openai
import sqlite3

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI & Notification Configurations
openai.api_key = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
ADMIN_PHONE_NUMBER = os.getenv("ADMIN_PHONE_NUMBER")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# ---------------------------------------
# Database Setup
# ---------------------------------------
DATABASE = "iot_data.db"

def init_db():
    """Initialize the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS iot_sensors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor TEXT,
            value REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS business_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            category TEXT,
            value REAL
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------------------------------
# Request Models
# ---------------------------------------
class PredictionRequest(BaseModel):
    category: str  # 'revenue', 'users', 'traffic'
    future_days: int
    model: str = "linear_regression"

class RecommendationRequest(BaseModel):
    category: str
    predicted_values: list

class IoTData(BaseModel):
    sensor: str
    value: float

# ---------------------------------------
# IoT Data Management
# ---------------------------------------
@app.post("/store-iot-data")
def store_iot_data(data: IoTData):
    """Store real-time IoT sensor data in the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    timestamp = datetime.datetime.utcnow().isoformat()
    cursor.execute("INSERT INTO iot_sensors (timestamp, sensor, value) VALUES (?, ?, ?)",
                   (timestamp, data.sensor, data.value))

    conn.commit()
    conn.close()
    return {"message": f"Stored {data.sensor} data: {data.value}"}

@app.get("/latest-iot-data")
def get_latest_iot_data():
    """Retrieve the latest IoT data from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT sensor, value FROM iot_sensors ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if row:
        return {"sensor": row[0], "value": row[1]}
    return {"error": "No IoT data available"}

# ---------------------------------------
# Business Question AI Answering
# ---------------------------------------
@app.post("/ask-question")
def ask_question(data: dict):
    """Handles business questions and generates AI-powered answers."""
    question = data.get("question", "")

    if not question:
        raise HTTPException(status_code=400, detail="No question provided")

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Answer this business question: {question}",
            max_tokens=100
        )
        answer = response["choices"][0]["text"].strip()
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------
# Business Data Connection
# ---------------------------------------
@app.post("/connect-data-source")
def connect_data_source(data: dict):
    """Simulates connecting a business data source."""
    name = data.get("name")
    data_type = data.get("type")
    path = data.get("path")

    if not all([name, data_type, path]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    return {"message": f"Connected {name} ({data_type}) at {path}"}

# ---------------------------------------
# AI-Generated Business Recommendations
# ---------------------------------------
@app.post("/generate-recommendations")
def generate_recommendations(request: RecommendationRequest):
    """Use OpenAI to generate business recommendations based on predicted values."""
    prompt = f"Based on the following predicted values for {request.category}: {request.predicted_values}, what are some strategic business recommendations?"
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return {"recommendations": response["choices"][0]["text"].strip()}
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------
# AI-Powered Anomaly Detection
# ---------------------------------------
@app.post("/detect-anomalies")
def detect_anomalies(data: dict):
    """Detect anomalies using Isolation Forest AI model."""
    category = data.get("category")
    if not category:
        raise HTTPException(status_code=400, detail="Category is required")

    # Simulated anomaly detection logic
    sample_data = np.array([random.uniform(5000, 20000) for _ in range(100)]).reshape(-1, 1)
    anomaly_detector = IsolationForest(n_estimators=100, contamination=0.05)
    anomaly_detector.fit(sample_data)

    latest_value = random.uniform(5000, 20000)
    is_anomaly = anomaly_detector.predict([[latest_value]])[0] == -1
    anomaly_score = random.uniform(0, 1)  # Simulating anomaly confidence score

    return {
        "anomalies": [
            {
                "value": latest_value,
                "score": anomaly_score,
                "is_anomaly": is_anomaly
            }
        ]
    }

# ---------------------------------------
# AI Dashboard Data Endpoint
# ---------------------------------------
@app.post("/ai-dashboard")
def ai_dashboard(data: dict):
    """Fetch AI-powered business metrics."""
    return {
        "revenue": round(random.uniform(50000, 150000), 2),
        "users": random.randint(1000, 5000),
        "traffic": random.randint(50000, 200000)
    }

# ---------------------------------------
# Run the App
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
