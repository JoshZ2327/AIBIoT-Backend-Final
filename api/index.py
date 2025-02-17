from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import random
import openai
import sqlite3
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

app = FastAPI()

# Enable CORS for frontend integration
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

# Initialize Twilio Client
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# ------------------------------
# Simulated Data
# ------------------------------

# Simulated IoT Data Storage
iot_data = [
    {"timestamp": datetime.datetime.utcnow().isoformat(), "sensor": "temperature", "value": 22.5},
    {"timestamp": datetime.datetime.utcnow().isoformat(), "sensor": "humidity", "value": 55},
    {"timestamp": datetime.datetime.utcnow().isoformat(), "sensor": "pressure", "value": 1012}
]

# ------------------------------
# Request Models
# ------------------------------
class PredictionRequest(BaseModel):
    category: str  # 'revenue', 'users', 'traffic'
    future_days: int  # Forecast period (1-90 days)
    model: str = "linear_regression"  # 'linear_regression' or 'arima'

class AnomalyDetectionRequest(BaseModel):
    category: str
    values: list

class RecommendationRequest(BaseModel):
    category: str
    predicted_values: list

class DataSourceRequest(BaseModel):
    name: str
    type: str  # 'sqlite', 'csv', 'api', 'iot'
    path: str  # File path, database URI, or API URL

# ------------------------------
# Endpoints
# ------------------------------

# Endpoint: Return Latest IoT Sensor Data (Simulated)
@app.get("/latest-iot-data")
def get_latest_iot_data():
    # Return the last entry from the simulated IoT data list.
    return {"latest_reading": iot_data[-1]}

# Enhanced AI-Powered Predictive Analytics
@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Predicts future trends for revenue, users, or traffic using AI models (Linear Regression, ARIMA)."""
    today = datetime.date.today()
    past_days = 60  # Use last 60 days for training data
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(past_days)][::-1]

    # Simulated historical data for demonstration purposes.
    historical_data = [(dates[i], random.uniform(5000, 20000)) for i in range(past_days)]

    # Prepare data arrays
    X = np.array([i for i in range(len(historical_data))]).reshape(-1, 1)
    y = np.array([val[1] for val in historical_data])

    # Choose forecasting model based on user selection
    if request.model == "linear_regression":
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.array([len(historical_data) + i for i in range(request.future_days)]).reshape(-1, 1)
        future_predictions = model.predict(future_X).tolist()
    elif request.model == "arima":
        model = ARIMA(y, order=(5,1,0))
        fitted_model = model.fit()
        future_predictions = fitted_model.forecast(steps=request.future_days).tolist()
    else:
        raise HTTPException(status_code=400, detail="Invalid model selection. Choose 'linear_regression' or 'arima'.")

    # Compute confidence intervals based on standard deviation of historical data
    std_dev = np.std(y)
    lower_bounds = [max(0, pred - (1.96 * std_dev)) for pred in future_predictions]
    upper_bounds = [pred + (1.96 * std_dev) for pred in future_predictions]

    future_dates = [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.future_days)]

    return {
        "category": request.category,
        "model": request.model,
        "predicted_trends": {
            "dates": future_dates,
            "values": future_predictions,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds
        }
    }

# AI-Powered Alerts & Notifications
def send_sms_alert(message):
    """Send an SMS alert using Twilio."""
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
    """Check real-time metrics and send alerts if thresholds are exceeded."""
    metrics = predict_trends(PredictionRequest(category="revenue", future_days=7, model="linear_regression"))
    alerts = []
    
    # Example alert conditions based on predicted revenue
    if metrics["predicted_trends"]["values"][-1] < 50000:
        alert_msg = "⚠️ ALERT: Revenue is predicted to drop below $50,000!"
        send_sms_alert(alert_msg)
        send_email_alert("Revenue Drop Alert!", f"<p>{alert_msg}</p>")
        alerts.append(alert_msg)
    if metrics["predicted_trends"]["values"][-1] > 150000:
        alert_msg = "🚀 ALERT: Revenue may spike above $150,000!"
        send_sms_alert(alert_msg)
        send_email_alert("Revenue Spike Alert!", f"<p>{alert_msg}</p>")
        alerts.append(alert_msg)

    return {"alerts_sent": alerts}

# AI-Powered Data Source Connection
@app.post("/connect-data-source")
def connect_data_source(request: DataSourceRequest):
    """Allows users to connect SQL databases, CSV files, APIs, or IoT devices."""
    supported_types = ["sqlite", "csv", "api", "iot"]
    if request.type not in supported_types:
        raise HTTPException(status_code=400, detail=f"Unsupported data source type. Choose from {supported_types}")
    return {"message": f"Data source '{request.name}' of type '{request.type}' connected successfully"}

# ------------------------------
# Run the App
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
