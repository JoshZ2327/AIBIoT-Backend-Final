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

# Request Models
class PredictionRequest(BaseModel):
    category: str  # 'revenue', 'users', 'traffic'
    future_days: int  # Forecast period (1-90 days)
    model: str = "linear_regression"  # User-selected model ('linear_regression' or 'arima')

class AnomalyDetectionRequest(BaseModel):
    category: str
    values: list

class RecommendationRequest(BaseModel):
    category: str
    predicted_values: list

# 🚀 AI-Powered Predictive Analytics
@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Predicts future trends for revenue, users, or traffic using AI models (Linear Regression, ARIMA)."""

    today = datetime.date.today()
    past_days = 60  # Use last 60 days for predictions
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(past_days)][::-1]

    # Simulated historical data
    historical_data = [(dates[i], random.uniform(5000, 20000)) for i in range(past_days)]

    # Step 1️⃣: Prepare Data
    X = np.array([i for i in range(len(historical_data))]).reshape(-1, 1)
    y = np.array([val[1] for val in historical_data])

    # Step 2️⃣: Train AI Model Based on User Selection
    if request.model == "linear_regression":
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.array([len(historical_data) + i for i in range(request.future_days)]).reshape(-1, 1)
        future_predictions = model.predict(future_X).tolist()

    elif request.model == "arima":
        model = ARIMA(y, order=(5,1,0))  # ARIMA (5,1,0) model
        fitted_model = model.fit()
        future_predictions = fitted_model.forecast(steps=request.future_days).tolist()

    else:
        raise HTTPException(status_code=400, detail="Invalid model selection. Choose 'linear_regression' or 'arima'.")

    # Step 3️⃣: Compute Confidence Intervals
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

# 🚀 AI-Powered Alerts & Notifications
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
    
    if metrics["predicted_trends"]["values"][-1] < 50000:
        alert_msg = f"⚠️ ALERT: Revenue is predicted to drop below $50,000!"
        send_sms_alert(alert_msg)
        send_email_alert("Revenue Drop Alert!", f"<p>{alert_msg}</p>")
        alerts.append(alert_msg)

    if metrics["predicted_trends"]["values"][-1] > 150000:
        alert_msg = f"🚀 ALERT: Revenue may spike above $150,000!"
        send_sms_alert(alert_msg)
        send_email_alert("Revenue Spike Alert!", f"<p>{alert_msg}</p>")
        alerts.append(alert_msg)

    return {"alerts_sent": alerts}

# ✅ Run the App
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
