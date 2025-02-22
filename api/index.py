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
import requests

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
# 🚀 Ask Business Questions (AI-Powered Answers)
# ---------------------------------------
@app.post("/ask-question")
def ask_question(data: AskQuestionRequest):
    """Handles business questions by integrating AI with connected data sources."""
    
    question = data.question
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")

    # 🔍 1️⃣ Retrieve connected data sources
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, type, path FROM data_sources")
    sources = cursor.fetchall()
    conn.close()

    # 🏢 2️⃣ If no data sources exist, fallback to AI-only answer
    if not sources:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Answer this business question: {question}",
            max_tokens=100
        )
        return {"answer": response["choices"][0]["text"].strip()}

    # 📊 3️⃣ Query each data source (example for SQLite, CSV, API)
    business_data = []
    for source in sources:
        name, data_type, path = source
        try:
            if data_type == "sqlite":
                conn = sqlite3.connect(path)
                df = pd.read_sql_query("SELECT * FROM business_metrics ORDER BY timestamp DESC LIMIT 10", conn)
                conn.close()
                business_data.append(f"Data from {name}:\n{df.to_string()}")
            elif data_type == "csv":
                df = pd.read_csv(path)
                business_data.append(f"Data from {name}:\n{df.head(10).to_string()}")
            elif data_type == "api":
                response = requests.get(path)
                business_data.append(f"Data from {name} API:\n{response.text[:500]}")
        except Exception as e:
            business_data.append(f"Could not retrieve data from {name}: {str(e)}")

    # 🔗 4️⃣ Combine Business Data with AI Question
    business_context = "\n\n".join(business_data)
    prompt = f"Based on the following business data:\n{business_context}\n\nAnswer this question: {question}"

    # 🚀 5️⃣ Generate AI-Powered Answer
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return {"answer": response["choices"][0]["text"].strip()}

# ---------------------------------------
# 🚀 AI Dashboard: Business Metrics & Predictions
# ---------------------------------------
@app.post("/ai-dashboard")
def ai_dashboard(data: dict):
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

    model = {"linear_regression": LinearRegression(),
             "arima": ARIMA(y, order=(5,1,0)),
             "prophet": Prophet()}.get(request.model)

    if not model:
        raise HTTPException(status_code=400, detail="Invalid model selection")

    model.fit(X, y)
    future_X = np.array([len(historical_data) + i for i in range(request.future_days)]).reshape(-1, 1)
    future_predictions = model.predict(future_X).tolist()

    return {"category": request.category, "predicted_values": future_predictions}

# ---------------------------------------
# 🚀 AI Alerts & Notifications (Every 5 min)
# ---------------------------------------
async def scheduled_ai_alerts():
    """Run AI trend detection every 5 minutes & send email alerts."""
    while True:
        new_data = pd.read_csv(CLOUD_STORAGE_PATH)
        predictions = new_data.mean(axis=0).to_dict()

        if check_significant_trends(predictions):
            summary = generate_summary(predictions)
            chart_path = generate_visual_chart(predictions)
            send_email_alert("🚨 Significant Business Trend Shift!", summary, chart_path)

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
