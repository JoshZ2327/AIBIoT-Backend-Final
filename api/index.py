from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import random
import os
import numpy as np
import pandas as pd
import asyncio
import smtplib
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import openai
import sqlite3

app = FastAPI()

# 🌍 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 AI & Notification Configurations
openai.api_key = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
ADMIN_PHONE_NUMBER = os.getenv("ADMIN_PHONE_NUMBER")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

SMTP_SERVER = "smtp.your-email.com"
SMTP_PORT = 587
SMTP_USERNAME = "your-email@example.com"
SMTP_PASSWORD = "your-password"
ALERT_RECIPIENTS = ["admin@example.com", "team@example.com"]

# 📊 Cloud Storage Simulation
CLOUD_STORAGE_PATH = "data/business_data.csv"

# ---------------------------------------
# 📀 Database Setup
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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT,
            path TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------------------------------
# Request Models
# ---------------------------------------
class PredictionRequest(BaseModel):
    category: str
    future_days: int
    model: str = "linear_regression"

class RecommendationRequest(BaseModel):
    category: str
    predicted_values: list

class IoTData(BaseModel):
    sensor: str
    value: float

class DataSource(BaseModel):
    name: str
    type: str
    path: str

class AskQuestionRequest(BaseModel):
    question: str

# ---------------------------------------
# 🚀 AI Dashboard: Business Metrics & Predictions
# ---------------------------------------
@app.post("/ai-dashboard")
def ai_dashboard():
    """Fetch AI-powered business metrics."""
    return {
        "revenue": round(random.uniform(50000, 150000), 2),
        "users": random.randint(1000, 5000),
        "traffic": random.randint(50000, 200000)
    }

@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Predict future trends using AI models."""
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
# 🚀 AI-Powered Business Recommendations
# ---------------------------------------
@app.post("/generate-recommendations")
def generate_recommendations(request: RecommendationRequest):
    """Use AI to generate business recommendations based on predicted trends."""
    prompt = f"Based on the predicted values for {request.category}: {request.predicted_values}, what are the best business decisions to make?"

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return {"recommendations": response["choices"][0]["text"].strip()}
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------
# 🚀 AI Alerts & Notifications (Every 5 min)
# ---------------------------------------
async def scheduled_ai_alerts():
    """Run AI trend detection every 5 minutes & send email alerts."""
    while True:
        print(f"🔄 Running AI trend detection at {datetime.datetime.utcnow()}...")

        # 🚀 Load Cloud Data
        new_data = pd.read_csv(CLOUD_STORAGE_PATH)

        # 📊 Run AI Forecasting
        predictions = new_data.mean(axis=0).to_dict()

        # 🚨 Detect Major Trend Shifts
        if check_significant_trends(predictions):
            summary = generate_summary(predictions)
            chart_path = generate_visual_chart(predictions)
            send_email_alert("🚨 Significant Business Trend Shift!", summary, chart_path)

        print(f"✅ AI trend analysis complete! Next run in 5 minutes.")
        await asyncio.sleep(300)

@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(scheduled_ai_alerts())

# ---------------------------------------
# 🚀 Run the App
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
