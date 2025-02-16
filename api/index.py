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

# AI Configuration (Replace with your OpenAI API key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Twilio & SendGrid Configuration
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
ADMIN_PHONE_NUMBER = os.getenv("ADMIN_PHONE_NUMBER")

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

# Initialize Twilio Client
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Simulated IoT Data Storage
iot_data = [
    {"timestamp": datetime.datetime.utcnow(), "sensor": "temperature", "value": 22.5},
    {"timestamp": datetime.datetime.utcnow(), "sensor": "humidity", "value": 55},
    {"timestamp": datetime.datetime.utcnow(), "sensor": "pressure", "value": 1012}
]

# Simulated Spare Parts Inventory
spare_parts_inventory = {
    "motor_bearing": {"current_stock": 5, "min_stock": 3, "supplier": "Supplier A"},
    "sensor_chip": {"current_stock": 8, "min_stock": 5, "supplier": "Supplier B"},
}

# Store connected data sources
data_sources = {}

# Request Models
class MaintenanceRequest(BaseModel):
    sensor_type: str

class RestockOrderRequest(BaseModel):
    part_name: str

class DataSourceRequest(BaseModel):
    name: str
    type: str
    path: str

class BusinessQuestion(BaseModel):
    question: str

class DashboardRequest(BaseModel):
    dateRange: int
    category: str

class PredictionRequest(BaseModel):
    category: str
    future_days: int

class AnomalyDetectionRequest(BaseModel):
    category: str
    values: list

class RecommendationRequest(BaseModel):
    category: str
    predicted_values: list

# 🚀 Send SMS Alert
def send_sms_alert(message):
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=ADMIN_PHONE_NUMBER
        )
        return {"message": "SMS Alert Sent!"}
    except Exception as e:
        return {"error": str(e)}

# 📧 Send Email Alert
def send_email_alert(subject, content):
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

### ✅ AI-Powered Alerts & Notifications ###
@app.post("/check-alerts")
def check_alerts():
    """Check real-time metrics and send alerts if thresholds are exceeded."""
    
    metrics = ai_dashboard(DashboardRequest(dateRange=7, category="all"))
    
    alerts = []
    
    if metrics["revenue"] < 50000:
        alert_msg = f"⚠️ ALERT: Revenue has dropped to ${metrics['revenue']}!"
        send_sms_alert(alert_msg)
        send_email_alert("Revenue Drop Alert!", f"<p>{alert_msg}</p>")
        alerts.append(alert_msg)

    if metrics["traffic"] > 50000:
        alert_msg = f"🚀 ALERT: Website traffic spike detected: {metrics['traffic']} visits!"
        send_sms_alert(alert_msg)
        send_email_alert("Traffic Spike Alert!", f"<p>{alert_msg}</p>")
        alerts.append(alert_msg)

    inventory_status = check_inventory()
    if inventory_status["restock_recommendations"]:
        for item in inventory_status["restock_recommendations"]:
            alert_msg = f"⚠️ ALERT: Low stock for {item['part_name']}! Current stock: {item['current_stock']}."
            send_sms_alert(alert_msg)
            send_email_alert("Low Inventory Alert!", f"<p>{alert_msg}</p>")
            alerts.append(alert_msg)

    return {"alerts_sent": alerts}

### ✅ Run the App ###
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
