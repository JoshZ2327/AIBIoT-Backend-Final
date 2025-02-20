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
# Predictive Analytics (Better Models)
# ---------------------------------------
@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Predict future trends using enhanced AI models."""
    today = datetime.date.today()
    past_days = 60
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(past_days)][::-1]

    historical_data = [(dates[i], random.uniform(5000, 20000)) for i in range(past_days)]
    X = np.array([i for i in range(len(historical_data))]).reshape(-1, 1)
    y = np.array([val[1] for val in historical_data])

    if request.model == "linear_regression":
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.array([len(historical_data) + i for i in range(request.future_days)]).reshape(-1, 1)
        future_predictions = model.predict(future_X).tolist()
    
    elif request.model == "arima":
        model = ARIMA(y, order=(5,1,0))
        fitted_model = model.fit()
        future_predictions = fitted_model.forecast(steps=request.future_days).tolist()
    
    elif request.model == "prophet":
        df = pd.DataFrame({"ds": dates, "y": y})
        prophet_model = Prophet()
        prophet_model.fit(df)
        future_df = pd.DataFrame({"ds": [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.future_days)]})
        forecast = prophet_model.predict(future_df)
        future_predictions = forecast["yhat"].tolist()
    
    else:
        raise HTTPException(status_code=400, detail="Invalid model selection")

    return {"category": request.category, "predicted_values": future_predictions}

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
# Alerts & Notifications
# ---------------------------------------
def send_sms_alert(message):
    """Send an SMS alert."""
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=ADMIN_PHONE_NUMBER
        )
        return {"message": "SMS Alert Sent!"}
    except Exception as e:
        return {"error": str(e)}

def send_email_alert(subject, content):
    """Send an email alert using SendGrid."""
    try:
        message = Mail(
            from_email="alerts@aibiot.com",
            to_emails=ADMIN_EMAIL,
            subject=subject,
            html_content=content
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        return {"message": "Email Alert Sent!"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/check-alerts")
def check_alerts():
    """Check for anomalies and send alerts."""
    anomaly_detector = IsolationForest(n_estimators=100, contamination=0.05)
    sample_data = np.array([random.uniform(5000, 20000) for _ in range(100)]).reshape(-1, 1)
    anomaly_detector.fit(sample_data)

    latest_data = random.uniform(5000, 20000)
    is_anomaly = anomaly_detector.predict([[latest_data]])[0] == -1

    if is_anomaly:
        alert_msg = f"⚠️ Anomaly detected: {latest_data}"
        send_sms_alert(alert_msg)
        send_email_alert("Anomaly Alert!", f"<p>{alert_msg}</p>")
        return {"alert": alert_msg}

    return {"message": "No anomalies detected"}

# ---------------------------------------
# Run the App
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
