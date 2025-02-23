from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import datetime
import random
import os
import numpy as np
import pandas as pd
import asyncio
import sqlite3
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from fastapi.middleware.cors import CORSMiddleware
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import openai

app = FastAPI()

# 🌍 Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# 🔑 AI & Notification Configurations
openai.api_key = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

# 📀 Database Setup
DATABASE = "ai_data.db"

def init_db():
    """Initialize the database with necessary tables."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT,
            path TEXT
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
        CREATE TABLE IF NOT EXISTS iot_sensors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor TEXT,
            value REAL
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------------------------------
# 📌 Data Models
# ---------------------------------------
class DataSource(BaseModel):
    name: str
    type: str
    path: str

class IoTData(BaseModel):
    sensor: str
    value: float

class PredictionRequest(BaseModel):
    category: str
    future_days: int
    model: str = "linear_regression"

class BusinessQuestion(BaseModel):
    question: str

# ---------------------------------------
# 🚀 AI-Powered Business Insights & Metrics
# ---------------------------------------

@app.post("/ai-dashboard")
def ai_dashboard():
    """Fetch AI-powered business metrics dynamically."""
    return {
        "revenue": round(random.uniform(50000, 150000), 2),  # Simulated revenue data
        "users": random.randint(1000, 5000),  # Simulated user count
        "traffic": random.randint(50000, 200000)  # Simulated website traffic
    }

# ---------------------------------------
# 🚀 AI-Powered Business Question Answering
# ---------------------------------------

CACHE = {}  # Global cache for frequently asked questions

def choose_best_cloud_provider():
    """Dynamically choose the cheapest cloud AI provider."""
    provider_costs = {
        "OpenAI": 0.02,  # Cost per 1,000 tokens (example)
        "AWS Bedrock": 0.015,
        "Google Vertex AI": 0.018
    }
    return min(provider_costs, key=provider_costs.get)  # Choose the lowest-cost provider

@app.post("/ask-question")
def ask_question(data: BusinessQuestion):
    """AI-powered Q&A with real-time business data integration, caching, and cost optimization."""

    question = data.question
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")

    # ✅ Step 1: Check cache to avoid redundant AI costs
    if question in CACHE:
        return {"answer": CACHE[question]}  # Return cached answer ✅

    # 🔍 Step 2: Retrieve Business Data
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, type, path FROM data_sources")
    sources = cursor.fetchall()
    conn.close()

    business_data = []
    for name, data_type, path in sources:
        if data_type == "sqlite":
            try:
                conn = sqlite3.connect(path)
                df = pd.read_sql_query("SELECT * FROM business_metrics ORDER BY timestamp DESC LIMIT 10", conn)
                conn.close()
                business_data.append(f"Data from {name}:\n{df.to_string()}")
            except Exception:
                pass

    business_context = "\n\n".join(business_data)
    prompt = f"Using this data:\n{business_context}\n\nAnswer: {question}"

    # ✅ Step 3: Choose the lowest-cost AI provider dynamically
    best_provider = choose_best_cloud_provider()
    
    if best_provider == "OpenAI":
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=200)
        answer = response["choices"][0]["text"].strip()
    elif best_provider == "AWS Bedrock":
        response = requests.post("https://aws-bedrock-endpoint", json={"text": prompt})
        answer = response.json().get("answer", "No answer available.")
    elif best_provider == "Google Vertex AI":
        response = requests.post("https://google-vertex-ai-endpoint", json={"text": prompt})
        answer = response.json().get("answer", "No answer available.")
    else:
        answer = "No suitable AI provider found."

    # ✅ Step 4: Store the result in cache to save costs
    CACHE[question] = answer  

    return {"answer": answer}
    
# ---------------------------------------
# 🚀 AI-Powered Alerts & WebSockets
# ---------------------------------------
alert_connections = []

@app.post("/check-alerts")
def check_alerts():
    """AI-generated alerts for business anomalies."""
    alerts = [
        {"category": "Sales", "message": "🚨 Sales dropped 20% in last 7 days!"},
        {"category": "Traffic", "message": "⚠️ Unusual website traffic detected."}
    ]
    return {"alerts": alerts}

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket for real-time alerts."""
    await websocket.accept()
    alert_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection open
    except WebSocketDisconnect:
        alert_connections.remove(websocket)

async def notify_alert_clients():
    """Broadcast AI-generated alerts."""
    alerts = check_alerts()
    disconnected_clients = []
    
    for connection in alert_connections:
        try:
            await connection.send_json(alerts)
        except:
            disconnected_clients.append(connection)

    for client in disconnected_clients:
        alert_connections.remove(client)

# ---------------------------------------
# 🚀 AI-Powered Predictive Analytics
# ---------------------------------------
@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Optimized AI Model Selection for Cost Efficiency"""

    today = datetime.date.today()
    past_days = 60
    y = np.array([random.uniform(5000, 20000) for _ in range(past_days)])

    if request.future_days <= 7:
        model = LinearRegression()
        model.fit(np.arange(past_days).reshape(-1, 1), y)
        future_predictions = model.predict(np.arange(past_days, past_days + request.future_days).reshape(-1, 1)).tolist()
    elif request.future_days <= 30:
        model = ARIMA(y, order=(5,1,0))
        fitted_model = model.fit()
        future_predictions = fitted_model.forecast(steps=request.future_days).tolist()
    else:
        df = pd.DataFrame({"ds": [str(today - datetime.timedelta(days=i)) for i in range(past_days)], "y": y})
        prophet_model = Prophet()
        prophet_model.fit(df)
        future_df = pd.DataFrame({"ds": [str(today + datetime.timedelta(days=i)) for i in range(request.future_days)]})
        forecast = prophet_model.predict(future_df)
        future_predictions = forecast["yhat"].tolist()

    return {"category": request.category, "predicted_values": future_predictions}

# ---------------------------------------
# 🚀 AI-Powered Batch Processing
# ---------------------------------------
async def run_ai_task(task):
    """Executes an AI processing task with optimized AI selection."""
    best_provider = choose_best_cloud_provider()

    if best_provider == "OpenAI":
        response = openai.Completion.create(engine="text-davinci-003", prompt=task, max_tokens=200)
        return response["choices"][0]["text"].strip()
    else:
        response = requests.post(f"https://{best_provider.lower()}-endpoint", json={"text": task})
        return response.json().get("answer", "No answer available.")

async def batch_process_ai_tasks(tasks):
    """Batch AI queries instead of running them one by one."""
    batch_size = 5
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        await asyncio.gather(*[run_ai_task(task) for task in batch])

@app.post("/batch-process-questions")
async def batch_process_questions(questions: list):
    """Processes multiple AI questions in a batch."""
    await batch_process_ai_tasks(questions)
    return {"message": "Batch processing started"}

# ---------------------------------------
# 🚀 Data Storage Optimization
# ---------------------------------------
def move_old_data_to_cold_storage():
    """Moves older AI results to cheaper cold storage."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM business_metrics WHERE timestamp < date('now', '-30 days')")
    old_data = cursor.fetchall()

    if old_data:
        storage_api_url = os.getenv("COLD_STORAGE_API", "https://secure-storage-service.com/upload")
        response = requests.post(storage_api_url, json={"data": old_data})
        if response.status_code == 200:
            cursor.execute("DELETE FROM business_metrics WHERE timestamp < date('now', '-30 days')")
            conn.commit()

    conn.close()

@app.on_event("startup")
async def schedule_storage_optimization():
    """Move old data to cold storage daily."""
    while True:
        move_old_data_to_cold_storage()
        await asyncio.sleep(86400)

# ---------------------------------------
# 🚀 Run the App
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
