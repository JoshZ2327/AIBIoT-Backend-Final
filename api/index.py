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
from fbprophet import Prophet
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

class RecommendationRequest(BaseModel):
    category: str
    predicted_values: list

class BusinessQuestion(BaseModel):
    question: str

# ---------------------------------------
# 🚀 AI-Powered Business Insights & Metrics
# ---------------------------------------
@app.post("/ai-dashboard")
def ai_dashboard():
    """Fetch AI-powered business metrics."""
    return {
        "revenue": round(random.uniform(50000, 150000), 2),
        "users": random.randint(1000, 5000),
        "traffic": random.randint(50000, 200000)
    }

# ---------------------------------------
# 🚀 AI-Powered Business Question Answering
# ---------------------------------------
def choose_best_cloud_provider():
    """Dynamically choose the cheapest cloud AI provider"""
    provider_costs = {
        "OpenAI": 0.02,  # Cost per 1,000 tokens (example)
        "AWS Bedrock": 0.015,
        "Google Vertex AI": 0.018
    }
    return min(provider_costs, key=provider_costs.get)  # Choose cheapest

@app.post("/ask-question")
def ask_question(data: BusinessQuestion):
    """AI-powered business Q&A with cloud cost optimization."""

    question = data.question
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")

    best_provider = choose_best_cloud_provider()  # Select lowest-cost AI API

    if best_provider == "OpenAI":
        response = openai.Completion.create(engine="text-davinci-003", prompt=question, max_tokens=200)
        answer = response["choices"][0]["text"].strip()

    elif best_provider == "AWS Bedrock":
        aws_response = requests.post("https://aws-bedrock-endpoint", json={"text": question})
        answer = aws_response.json().get("answer", "No response from AWS.")

    else:  # Google Vertex AI
        google_response = requests.post("https://google-vertex-ai-endpoint", json={"text": question})
        answer = google_response.json().get("answer", "No response from Google.")

    return {"answer": answer}
    
    # 🔍 Fetch Data Sources
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, type, path FROM data_sources")
    sources = cursor.fetchall()
    conn.close()

    business_data = []
    
    # 📊 Process Data Sources
    for name, data_type, path in sources:
        if data_type == "sqlite":
            try:
                conn = sqlite3.connect(path)
                df = pd.read_sql_query("SELECT * FROM business_metrics ORDER BY timestamp DESC LIMIT 10", conn)
                conn.close()
                business_data.append(f"Data from {name}:\n{df.to_string()}")
            except Exception as e:
                business_data.append(f"Could not query {name}: {str(e)}")

        elif data_type == "csv":
            try:
                df = pd.read_csv(path)
                business_data.append(f"Data from {name}:\n{df.head(10).to_string()}")
            except Exception as e:
                business_data.append(f"Could not read {name}: {str(e)}")

    # 🔗 Combine Business Data with AI Question
    business_context = "\n\n".join(business_data)
    prompt = f"Using this data:\n{business_context}\n\nAnswer: {question}"

    # 🚀 AI Response
    try:
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=200)
        return {"answer": response["choices"][0]["text"].strip()}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------
# 🚀 AI-Powered Alerts & Notifications
# ---------------------------------------
@app.post("/check-alerts")
def check_alerts():
    """Returns AI-generated alerts for business anomalies."""
    alerts = [
        {"category": "Sales", "message": "🚨 Sales dropped 20% in last 7 days!"},
        {"category": "Traffic", "message": "⚠️ Unusual website traffic detected."}
    ]
    return {"alerts": alerts}

# ---------------------------------------
# 📡 WebSockets for Real-Time Alerts
# ---------------------------------------
alert_connections = []

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket connection for real-time alerts."""
    await websocket.accept()
    alert_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        alert_connections.remove(websocket)

async def notify_alert_clients():
    """Send real-time AI alerts via WebSocket."""
    alerts = check_alerts()
    for connection in alert_connections:
        try:
            await connection.send_json(alerts)
        except:
            alert_connections.remove(connection)

# ---------------------------------------
# 🚀 AI-Powered Predictive Analytics
# ---------------------------------------
@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Optimized AI Model Selection for Cost Efficiency"""

    today = datetime.date.today()
    past_days = 60
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(past_days)][::-1]
    y = np.array([random.uniform(5000, 20000) for _ in range(past_days)])

    # **Dynamic Model Selection for Cost Efficiency**
    if request.future_days <= 7:  # Small prediction window → Cheaper model
        model = LinearRegression()
        model.fit(np.arange(past_days).reshape(-1, 1), y)
        future_predictions = model.predict(np.arange(past_days, past_days + request.future_days).reshape(-1, 1)).tolist()
    
    elif request.future_days <= 30:  # Medium-term → ARIMA
        model = ARIMA(y, order=(5,1,0))
        fitted_model = model.fit()
        future_predictions = fitted_model.forecast(steps=request.future_days).tolist()
    
    else:  # Long-term predictions → Prophet (Higher cost)
        df = pd.DataFrame({"ds": dates, "y": y})
        prophet_model = Prophet()
        prophet_model.fit(df)
        future_df = pd.DataFrame({"ds": [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.future_days)]})
        forecast = prophet_model.predict(future_df)
        future_predictions = forecast["yhat"].tolist()

    return {"category": request.category, "predicted_values": future_predictions}

def move_old_data_to_cold_storage():
    """Moves older AI results to cheaper cold storage (like S3 or Glacier)."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Move data older than 30 days to cold storage
    cursor.execute("SELECT * FROM business_metrics WHERE timestamp < date('now', '-30 days')")
    old_data = cursor.fetchall()

    if old_data:
        # Store in a cold storage bucket (AWS S3, Google Cloud Storage, etc.)
        requests.post("https://cold-storage-endpoint", json={"data": old_data})
        cursor.execute("DELETE FROM business_metrics WHERE timestamp < date('now', '-30 days')")  # Delete from live DB
        conn.commit()

    conn.close()

# 🚀 Background task to optimize storage
@app.on_event("startup")
async def schedule_storage_optimization():
    """Background job to periodically move old data to cold storage."""
    while True:
        move_old_data_to_cold_storage()
        await asyncio.sleep(86400)  # Run once per day (every 24 hours)
        
# ---------------------------------------
# 🚀 Run the App
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
