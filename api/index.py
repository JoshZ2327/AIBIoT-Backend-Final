
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
# 🚀 API Endpoints for Managing Data Sources
# ---------------------------------------

@app.get("/list-data-sources")
def list_data_sources():
    """Fetch data sources from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, type, path FROM data_sources")
    sources = [{"name": row[0], "type": row[1], "path": row[2]} for row in cursor.fetchall()]
    conn.close()
    return {"sources": sources}

@app.delete("/delete-data-source/{name}")
def delete_data_source(name: str):
    """Delete a data source from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM data_sources WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    return {"message": f"Data source {name} deleted successfully."}
    
# ---------------------------------------
# 🚀 AI-Powered Business Insights & Metrics
# ---------------------------------------
@app.post("/ai-dashboard")
def ai_dashboard():
    """Fetch AI-powered business metrics from multiple sources."""

    metrics = {}

    # ✅ Fetch Metrics from SQLite
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT category, value FROM business_metrics ORDER BY timestamp DESC LIMIT 3")
    rows = cursor.fetchall()
    conn.close()
    
    for row in rows:
        metrics[row[0]] = row[1]

    # ✅ Fetch Metrics from MySQL
    try:
        import mysql.connector
        conn = mysql.connector.connect(host="your-host", user="your-user", password="your-pass", database="your-db")
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT category, value FROM business_metrics ORDER BY timestamp DESC LIMIT 3")
        rows = cursor.fetchall()
        for row in rows:
            metrics[row["category"]] = row["value"]
        conn.close()
    except Exception as e:
        print(f"MySQL Error: {e}")

    # ✅ Fetch Metrics from MongoDB
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb+srv://your-mongodb-url")
        db = client["your-db"]
        collection = db["business_metrics"]
        documents = collection.find().limit(3)
        for doc in documents:
            metrics[doc["category"]] = doc["value"]
    except Exception as e:
        print(f"MongoDB Error: {e}")

    return metrics

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
    """AI-powered Q&A with real-time business data from multiple sources, including APIs and IoT sensors."""

    question = data.question
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")

    business_data = []
    
    # ✅ Fetch Data from SQLite
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, type, path FROM data_sources")
    sources = cursor.fetchall()
    conn.close()

    for name, data_type, path in sources:
        if data_type == "sqlite":
            try:
                conn = sqlite3.connect(path)
                df = pd.read_sql_query("SELECT * FROM business_metrics ORDER BY timestamp DESC LIMIT 10", conn)
                conn.close()
                business_data.append(f"Data from {name}:\n{df.to_string()}")
            except Exception as e:
                print(f"SQLite Error: {e}")

        elif data_type == "csv":
            try:
                df = pd.read_csv(path)
                business_data.append(f"Data from {name}:\n{df.head(10).to_string()}")
            except Exception as e:
                print(f"CSV Error: {e}")

        elif data_type == "mysql":
            try:
                import mysql.connector
                conn = mysql.connector.connect(host="your-host", user="your-user", password="your-pass", database="your-db")
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM business_metrics ORDER BY timestamp DESC LIMIT 10")
                rows = cursor.fetchall()
                business_data.append(f"MySQL Data from {name}:\n{rows}")
                conn.close()
            except Exception as e:
                print(f"MySQL Error: {e}")

        elif data_type == "mongodb":
            try:
                from pymongo import MongoClient
                client = MongoClient("mongodb+srv://your-mongodb-url")
                db = client["your-db"]
                collection = db["business_metrics"]
                documents = collection.find().limit(10)
                business_data.append(f"MongoDB Data from {name}:\n{list(documents)}")
            except Exception as e:
                print(f"MongoDB Error: {e}")

    # ✅ Fetch IoT Data from MQTT Broker
    try:
        import paho.mqtt.client as mqtt

        def on_message(client, userdata, msg):
            business_data.append(f"IoT Sensor Data: {msg.topic} - {msg.payload.decode()}")

        mqtt_client = mqtt.Client()
        mqtt_client.on_message = on_message
        mqtt_client.connect("mqtt-broker-url", 1883, 60)
        mqtt_client.subscribe("iot/sensors/temperature")
        mqtt_client.loop_start()
        asyncio.sleep(2)  # Wait for messages
        mqtt_client.loop_stop()
    except Exception as e:
        print(f"IoT MQTT Error: {e}")

    # ✅ Fetch IoT Data from Firebase
    try:
        import firebase_admin
        from firebase_admin import credentials, db

        cred = credentials.Certificate("path-to-firebase-credentials.json")
        firebase_admin.initialize_app(cred, {"databaseURL": "https://your-firebase-url.firebaseio.com/"})

        ref = db.reference("/iot_sensors")
        iot_data = ref.order_by_child("timestamp").limit_to_last(5).get()

        if iot_data:
            business_data.append(f"Firebase IoT Data:\n{iot_data}")
    except Exception as e:
        print(f"Firebase IoT Error: {e}")

    # ✅ Fetch Data from External APIs
    api_endpoints = [
        "https://api.example.com/business-metrics",
        "https://api.example.com/user-analytics",
        "https://api.example.com/market-trends"
    ]

    for url in api_endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                business_data.append(f"API Data from {url}:\n{response.json()}")
            else:
                print(f"API Error: {url} - Status {response.status_code}")
        except Exception as e:
            print(f"API Request Failed: {url} - {e}")

    # ✅ Use AI to Generate Business Insights
    business_context = "\n\n".join(business_data)
    prompt = f"Using this data:\n{business_context}\n\nAnswer: {question}"

    # ✅ Choose the Lowest-Cost AI Provider
    try:
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=200)
        answer = response["choices"][0]["text"].strip()
    except Exception as e:
        answer = f"Error processing AI request: {str(e)}"

    return {"answer": answer}

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

def fetch_data_sources_from_db():
    """Fetches the latest list of data sources from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, type, path FROM data_sources")
    sources = [{"name": row[0], "type": row[1], "path": row[2]} for row in cursor.fetchall()]
    conn.close()
    return sources

def fetch_latest_iot_data():
    """Fetches the latest IoT sensor data from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, sensor, value FROM iot_sensors ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return {"timestamp": row[0], "sensor": row[1], "value": row[2]} if row else {}
    
@app.websocket("/ws/data-sources")
async def websocket_data_sources(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = {"type": "update", "data_sources": fetch_data_sources_from_db()}  
            await websocket.send_json(data)
            await asyncio.sleep(5)  # Send updates every 5 seconds
    except WebSocketDisconnect:
        print("❌ Data Sources WebSocket Disconnected")

@app.websocket("/ws/iot")
async def websocket_iot(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            latest_iot_data = fetch_latest_iot_data()  
            await websocket.send_json(latest_iot_data)
            await asyncio.sleep(5)  # Send updates every 5 seconds
    except WebSocketDisconnect:
        print("❌ IoT WebSocket Disconnected")
        
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
    
# ---------------------------------------
# 🚀 Data Storage Optimization
# ---------------------------------------

def move_old_data_to_cold_storage():
    """Moves older AI results to cheaper cold storage (like S3 or Glacier)."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Move data older than 30 days to cold storage
    cursor.execute("SELECT * FROM business_metrics WHERE timestamp < date('now', '-30 days')")
    old_data = cursor.fetchall()

    if old_data:
        # 🚀 Replace this placeholder with your actual cold storage API
        storage_api_url = os.getenv("COLD_STORAGE_API", "https://secure-storage-service.com/upload")
        
        response = requests.post(storage_api_url, json={"data": old_data})
        
        if response.status_code == 200:
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
# 🚀 Background task to optimize storage
@app.on_event("startup")
async def schedule_storage_optimization():
    """Background job to periodically move old data to cold storage."""
    while True:
        move_old_data_to_cold_storage()
        await asyncio.sleep(86400)  # Run once per day (every 24 hours)

# ---------------------------------------
# 🚀 AI-Powered Batch Processing for Cost Efficiency
# ---------------------------------------

# ---------------------------------------
# 🚀 AI-Powered Batch Processing for Cost Efficiency
# ---------------------------------------

async def run_ai_task(task):
    """Executes an AI processing task with optimized AI selection."""
    best_provider = choose_best_cloud_provider()

    if best_provider == "OpenAI":
        response = openai.Completion.create(engine="text-davinci-003", prompt=task, max_tokens=200)
        return response["choices"][0]["text"].strip()
    elif best_provider == "AWS Bedrock":
        response = requests.post("https://aws-bedrock-endpoint", json={"text": task})
        return response.json().get("answer", "No answer available.")
    elif best_provider == "Google Vertex AI":
        response = requests.post("https://google-vertex-ai-endpoint", json={"text": task})
        return response.json().get("answer", "No answer available.")
    else:
        return "No AI provider available."

async def batch_process_ai_tasks(tasks):
    """Groups AI queries into batches instead of running them one by one."""
    batch_size = 5  # Process 5 queries at a time to reduce API costs
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        await asyncio.gather(*[run_ai_task(task) for task in batch])

@app.post("/batch-process-questions")
async def batch_process_questions(questions: list):
    """Receives multiple questions and processes them in a batch."""
    await batch_process_ai_tasks(questions)
    return {"message": "Batch processing started"}
async def batch_process_ai_tasks(tasks):
    """Groups AI queries into batches instead of running them one by one."""
    batch_size = 5  # Process 5 queries at a time to reduce API costs
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        await asyncio.gather(*[run_ai_task(task) for task in batch])

async def run_ai_task(task):
    """Executes an AI processing task."""
    response = openai.Completion.create(engine="text-davinci-003", prompt=task, max_tokens=200)
    return response["choices"][0]["text"].strip()

@app.post("/batch-process-questions")
async def batch_process_questions(questions: list):
    """Receives multiple questions and processes them in a batch."""
    await batch_process_ai_tasks(questions)
    return {"message": "Batch processing started"}

# ---------------------------------------
# 🚀 Run the App
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
