from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import datetime
import random
import os
import numpy as np
import pandas as pd
import asyncio
import sqlite3
import smtplib
import requests
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
from fastapi.middleware.cors import CORSMiddleware
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import openai

app = FastAPI()

# 🌍 Enable CORS
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
# 🚀 AI-Powered Business Question Answering
# ---------------------------------------
@app.post("/ask-question")
def ask_question(data: BusinessQuestion):
    """AI-powered business Q&A with real-time data integration."""
    
    question = data.question
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")

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
# 🚀 AI-Powered Business Insights & Predictions
# ---------------------------------------
@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Predict business trends using AI models."""
    
    today = datetime.date.today()
    past_days = 60
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(past_days)][::-1]
    y = np.array([random.uniform(5000, 20000) for _ in range(past_days)])

    if request.model == "linear_regression":
        model = LinearRegression()
        model.fit(np.arange(past_days).reshape(-1, 1), y)
        future_predictions = model.predict(np.arange(past_days, past_days + request.future_days).reshape(-1, 1)).tolist()
    
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
# 📊 AI-Powered Alerts & Automation
# ---------------------------------------
@app.post("/get-recommendations")
def get_recommendations(data: RecommendationRequest):
    """AI-generated business recommendations based on predictions."""
    
    category = data.category
    recommendations = []

    if category == "sales":
        recommendations = ["Increase ad budget for trending products.", "Launch a discount campaign."]
    elif category == "traffic":
        recommendations = ["Optimize website speed.", "Invest in SEO for trending keywords."]
    elif category == "user growth":
        recommendations = ["Introduce referral bonuses.", "Enhance social media strategies."]

    return {"recommendations": recommendations}

# ---------------------------------------
# 📡 WebSockets for Real-Time Data Updates
# ---------------------------------------
active_connections = []

@app.websocket("/ws/data-sources")
async def websocket_data_sources(websocket: WebSocket):
    """WebSocket connection for real-time data source updates."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()  
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def notify_clients():
    """Notify WebSocket clients when data sources update."""
    data_sources = get_all_data_sources()
    for connection in active_connections:
        try:
            await connection.send_json({"data_sources": data_sources})
        except:
            active_connections.remove(connection)

def get_all_data_sources():
    """Fetch all data sources from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, type, path FROM data_sources")
    sources = [{"name": row[0], "type": row[1], "path": row[2]} for row in cursor.fetchall()]
    conn.close()
    return sources

@app.post("/connect-data-source")
async def connect_data_source(data: DataSource):
    """Registers a new data source and notifies WebSocket clients."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO data_sources (name, type, path) VALUES (?, ?, ?)", (data.name, data.type, data.path))
    conn.commit()
    conn.close()

    await notify_clients()
    return {"message": f"Data source {data.name} connected successfully."}

# ---------------------------------------
# 🚀 Run the App
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
