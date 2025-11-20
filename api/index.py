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
from routers import voice
app.include_router(voice.router)

app = FastAPI()
# ...
app.include_router(voice.router)

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

    # Table for Data Sources
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT,
            path TEXT
        )
    """)

    # Table for Business Metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS business_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            category TEXT,
            value REAL
        )
    """)

    # Table for IoT Sensor Data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS iot_sensors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor TEXT,
            value REAL
        )
    """)

    # Table for IoT Anomalies
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS iot_anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor TEXT,
            value REAL,
            anomaly_score REAL,
            status TEXT  -- 'High', 'Medium', 'Low'
        )
    """)
        # Table for AI Anomaly Explanations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_anomaly_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor TEXT,
            anomaly TEXT,
            explanation TEXT,
            recommended_fix TEXT,
            preventive_action TEXT
        )
    """)
    
    # Table for IoT Automation Rules
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS iot_automation_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_condition TEXT,
            action TEXT
        )
    """)

    # Table for IoT Automation Logs
cursor.execute("""
    CREATE TABLE IF NOT EXISTS iot_automation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        action TEXT
        )
    """)

    # Table for AI-Adjusted Thresholds
cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_adjusted_thresholds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sensor TEXT UNIQUE,
        baseline_threshold REAL,
        adjusted_threshold REAL,
        last_updated TEXT
    )
""")
    # ✅ Table for Voice Questions & Responses
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            response TEXT
        )
    """)
    # ✅ Digital Twin Table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS digital_twins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_name TEXT UNIQUE,
        asset_type TEXT,
        sensor_data TEXT,  -- JSON format: { "temperature": 22.5, "pressure": 30 }
        ai_thresholds TEXT,  -- JSON format: { "temperature": 25.0 }
        status TEXT DEFAULT 'Active',
        last_updated TEXT DEFAULT CURRENT_TIMESTAMP
    )
""")

    conn.commit()
    conn.close()
    
init_db()

from sklearn.ensemble import IsolationForest
import numpy as np

def detect_anomalies(sensor_name: str):
    """Detect anomalies in IoT sensor data using Isolation Forest and generate AI explanations."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Fetch last 100 readings for the sensor
    cursor.execute("SELECT timestamp, value FROM iot_sensors WHERE sensor = ? ORDER BY timestamp DESC LIMIT 100", (sensor_name,))
    data = cursor.fetchall()
    conn.close()
    
    if len(data) < 10:
        return []  # Not enough data to detect anomalies
    
    timestamps, values = zip(*data)
    values = np.array(values).reshape(-1, 1)  # Reshape for model input

    # Train Isolation Forest Model
    model = IsolationForest(contamination=0.05, random_state=42)  # Detects 5% of points as anomalies
    model.fit(values)
    anomaly_scores = model.decision_function(values)
    predictions = model.predict(values)

    anomalies = []
    for i in range(len(predictions)):
        if predictions[i] == -1:  # -1 indicates an anomaly
            status = "High" if anomaly_scores[i] < -0.1 else "Medium"
            anomaly_details = {
                "timestamp": timestamps[i],
                "sensor": sensor_name,
                "value": values[i][0],
                "anomaly_score": anomaly_scores[i],
                "status": status
            }
            anomalies.append(anomaly_details)

            # ✅ Generate AI Explanation for Detected Anomaly
            generate_ai_anomaly_explanation(sensor_name, anomaly_details)

    # Save anomalies to DB
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    for anomaly in anomalies:
        cursor.execute("INSERT INTO iot_anomalies (timestamp, sensor, value, anomaly_score, status) VALUES (?, ?, ?, ?, ?)", 
                       (anomaly["timestamp"], anomaly["sensor"], anomaly["value"], anomaly["anomaly_score"], anomaly["status"]))
    conn.commit()
    conn.close()

    return anomalies
import numpy as np
from sklearn.ensemble import IsolationForest

def adjust_thresholds(sensor_name: str):
    """AI dynamically adjusts IoT automation thresholds based on historical sensor data."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Fetch the last 100 readings for the sensor
    cursor.execute("SELECT timestamp, value FROM iot_sensors WHERE sensor = ? ORDER BY timestamp DESC LIMIT 100", (sensor_name,))
    data = cursor.fetchall()
    conn.close()

    if len(data) < 10:
        print(f"⚠️ Not enough data to adjust threshold for {sensor_name}.")
        return

    timestamps, values = zip(*data)
    values = np.array(values).reshape(-1, 1)  # Convert values for ML model

    # Train an Isolation Forest model to detect patterns
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(values)

    # Calculate the **95th percentile value** as the new threshold
    adjusted_threshold = np.percentile(values, 95)

    # Store new AI-adjusted threshold in database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Check if threshold already exists
    cursor.execute("SELECT * FROM ai_adjusted_thresholds WHERE sensor = ?", (sensor_name,))
    existing_entry = cursor.fetchone()

    if existing_entry:
        cursor.execute("UPDATE ai_adjusted_thresholds SET adjusted_threshold = ?, last_updated = datetime('now') WHERE sensor = ?", 
                       (adjusted_threshold, sensor_name))
    else:
        cursor.execute("INSERT INTO ai_adjusted_thresholds (sensor, baseline_threshold, adjusted_threshold, last_updated) VALUES (?, ?, ?, datetime('now'))", 
                       (sensor_name, np.mean(values), adjusted_threshold))

    conn.commit()
    conn.close()

    print(f"✅ AI adjusted threshold for {sensor_name}: {adjusted_threshold}")
    
def generate_ai_anomaly_explanation(sensor, anomaly_details):
    """Generates an AI-powered explanation for anomalies."""
    anomaly_value = anomaly_details["value"]
    anomaly_score = anomaly_details["anomaly_score"]
    status = anomaly_details["status"]

    # ✅ Construct OpenAI Prompt
    prompt = f"""
    The following IoT sensor has detected an anomaly:

    📡 **Sensor:** {sensor}
    ❌ **Detected Value:** {anomaly_value}
    📊 **Anomaly Score:** {anomaly_score}
    🚨 **Severity Level:** {status}

    Provide an **explanation** for why this anomaly occurred, a **recommended fix**, and a **preventive action**.

    Response Format:
    📌 Explanation:
    🔧 Recommended Fix:
    🔮 Preventive Action:
    """

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200
        )
        
        ai_text = response["choices"][0]["text"].strip().split("\n")

        # ✅ Extract AI-generated insights
        explanation = ai_text[0].replace("📌 Explanation:", "").strip()
        recommended_fix = ai_text[1].replace("🔧 Recommended Fix:", "").strip()
        preventive_action = ai_text[2].replace("🔮 Preventive Action:", "").strip()

        # ✅ Save to Database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO ai_anomaly_explanations (timestamp, sensor, anomaly, explanation, recommended_fix, preventive_action)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, sensor, f"Value: {anomaly_value}, Score: {anomaly_score}, Severity: {status}",
              explanation, recommended_fix, preventive_action))
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"❌ AI Anomaly Explanation Error: {e}")
        
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
# 🚀 IoT Automation API Endpoints
# ---------------------------------------

class AutomationRule(BaseModel):
    trigger_condition: str
    action: str

@app.post("/add-automation-rule")
def add_automation_rule(rule: AutomationRule):
    """Adds a new automation rule to the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO iot_automation_rules (trigger_condition, action) VALUES (?, ?)", 
                   (rule.trigger_condition, rule.action))
    conn.commit()
    conn.close()
    return {"message": "Automation rule added successfully."}

@app.get("/get-automation-rules")
def get_automation_rules():
    """Fetches all stored automation rules."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, trigger_condition, action FROM iot_automation_rules")
    rules = [{"id": row[0], "trigger_condition": row[1], "action": row[2]} for row in cursor.fetchall()]
    conn.close()
    return {"rules": rules}

@app.delete("/delete-automation-rule/{rule_id}")
def delete_automation_rule(rule_id: int):
    """Deletes a specific automation rule from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM iot_automation_rules WHERE id = ?", (rule_id,))
    conn.commit()
    conn.close()
    return {"message": "Automation rule deleted successfully."}
    
from fastapi import FastAPI, HTTPException
import json
import sqlite3

app = FastAPI()

DATABASE = "ai_data.db"

# 🚀 Digital Twin API Endpoints
@app.post("/add-digital-twin")
def add_digital_twin(asset_name: str, asset_type: str):
    """Create a new Digital Twin."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Default empty JSON structure for sensor data and thresholds
    sensor_data = json.dumps({})
    ai_thresholds = json.dumps({})

    try:
        cursor.execute("INSERT INTO digital_twins (asset_name, asset_type, sensor_data, ai_thresholds) VALUES (?, ?, ?, ?)", 
                       (asset_name, asset_type, sensor_data, ai_thresholds))
        conn.commit()
        return {"message": f"✅ Digital Twin '{asset_name}' added successfully!"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail=f"❌ Digital Twin '{asset_name}' already exists.")

@app.get("/get-digital-twins")
def get_digital_twins():
    """Fetch all Digital Twins."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT asset_name, asset_type, sensor_data, ai_thresholds, status FROM digital_twins")
    twins = [{"asset_name": row[0], "asset_type": row[1], "sensor_data": json.loads(row[2]), "ai_thresholds": json.loads(row[3]), "status": row[4]} for row in cursor.fetchall()]
    conn.close()
    return {"digital_twins": twins}

@app.post("/update-digital-twin")
def update_digital_twin(asset_name: str, sensor_data: dict):
    """Update sensor data of a Digital Twin."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Convert sensor data to JSON format
    sensor_data_json = json.dumps(sensor_data)

    cursor.execute("UPDATE digital_twins SET sensor_data = ?, last_updated = CURRENT_TIMESTAMP WHERE asset_name = ?", 
                   (sensor_data_json, asset_name))
    conn.commit()
    conn.close()
    return {"message": f"✅ Digital Twin '{asset_name}' updated successfully!"}
@app.get("/get-digital-twin-3d")
def get_digital_twin_3d(asset_name: str):
    """Fetch 3D visualization data for a Digital Twin."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT asset_name, sensor_data, ai_thresholds FROM digital_twins WHERE asset_name = ?", (asset_name,))
    row = cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail=f"❌ Digital Twin '{asset_name}' not found.")

    asset_data = {
        "asset_name": row[0],
        "sensor_data": json.loads(row[1]),
        "ai_thresholds": json.loads(row[2]),
    }

    conn.close()
    return asset_data
    
import json
import sqlite3
import numpy as np
from sklearn.ensemble import IsolationForest

DATABASE = "ai_data.db"

# 🚀 AI-Adjusted IoT Automation Thresholds for Digital Twins
def adjust_thresholds(asset_name: str):
    """AI dynamically adjusts IoT automation thresholds based on Digital Twin data."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT sensor_data FROM digital_twins WHERE asset_name = ?", (asset_name,))
    row = cursor.fetchone()

    if not row:
        print(f"❌ Digital Twin '{asset_name}' not found.")
        return

    sensor_data = json.loads(row[0])
    values = np.array(list(sensor_data.values())).reshape(-1, 1)

    if len(values) < 10:
        print(f"⚠️ Not enough data to adjust threshold for {asset_name}.")
        return

    # Train AI model to detect patterns
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(values)

    # Adjust threshold to 95th percentile of historical sensor data
    adjusted_threshold = np.percentile(values, 95)
    ai_thresholds = json.dumps({"adjusted_threshold": adjusted_threshold})

    cursor.execute("UPDATE digital_twins SET ai_thresholds = ?, last_updated = CURRENT_TIMESTAMP WHERE asset_name = ?", 
                   (ai_thresholds, asset_name))
    conn.commit()
    conn.close()

    print(f"✅ AI adjusted threshold for {asset_name}: {adjusted_threshold}")

# ✅ Example API Endpoint to Trigger AI Threshold Adjustments
@app.post("/adjust-ai-thresholds")
def adjust_ai_thresholds(asset_name: str):
    """API endpoint to trigger AI threshold adjustments for a Digital Twin."""
    adjust_thresholds(asset_name)
    return {"message": f"✅ AI thresholds adjusted for {asset_name}."}
    
# ---------------------------------------
# 🚀 Voice Command Processing for IoT Control
# ---------------------------------------

class VoiceCommandRequest(BaseModel):
    command: str

@app.post("/voice-command")
def process_voice_command(request: VoiceCommandRequest):
    """Processes voice commands and triggers IoT actions."""
    command = request.command.lower()

    # ✅ Define voice-triggered actions
    action_mapping = {
        "turn on the lights": "Turn on Lights",
        "turn off the lights": "Turn off Lights",
        "increase temperature": "Increase Temperature",
        "decrease temperature": "Decrease Temperature",
        "open the door": "Open Door",
        "close the door": "Close Door",
        "activate alarm": "Activate Alarm",
        "deactivate alarm": "Deactivate Alarm"
    }

    matched_action = None
    for phrase, action in action_mapping.items():
        if phrase in command:
            matched_action = action
            break

    if matched_action:
        # ✅ Log action in IoT Automation Logs
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO iot_automation_logs (timestamp, action) VALUES (?, ?)", (timestamp, matched_action))
        conn.commit()
        conn.close()

        return {"message": f"IoT Action Executed: {matched_action}"}

    return {"message": "❌ Command not recognized"}
    
@app.get("/get-automation-logs")
def get_automation_logs():
    """Fetches all executed automation actions."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, action FROM iot_automation_logs ORDER BY timestamp DESC LIMIT 50")
    logs = [{"timestamp": row[0], "action": row[1]} for row in cursor.fetchall()]
    conn.close()
    return {"logs": logs}
    
import ast

def safe_eval_condition(condition: str, sensor_value: float):
    """Safely evaluates automation rule conditions."""
    condition = condition.replace("Temperature", str(sensor_value))

    try:
        # ✅ Securely parse the condition and evaluate
        node = ast.parse(condition, mode='eval')
        if isinstance(node, ast.Expression):
            return eval(compile(node, "<string>", "eval"))
    except Exception as e:
        print(f"❌ Error evaluating automation rule: {e}")
        return False

def check_automation_rules(iot_data):
    """Checks if any AI-adjusted automation rule matches the incoming IoT data."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Fetch AI-adjusted threshold for this sensor
    cursor.execute("SELECT adjusted_threshold FROM ai_adjusted_thresholds WHERE sensor = ?", (iot_data["sensor"],))
    adjusted_threshold = cursor.fetchone()
    
    if adjusted_threshold:
        threshold_value = adjusted_threshold[0]  # AI-determined threshold
    else:
        return []  # No AI threshold available, skip automation

    # Fetch automation rules
    cursor.execute("SELECT trigger_condition, action FROM iot_automation_rules")
    rules = cursor.fetchall()
    conn.close()

    triggered_actions = []
    for rule in rules:
        trigger_condition = rule[0].replace("THRESHOLD", str(threshold_value))  # Replace placeholder
        action = rule[1]

        # ✅ Check if IoT value exceeds AI threshold
        if eval(trigger_condition):
            triggered_actions.append(action)
            print(f"🔥 AI-Triggered Automation Rule: {action}")

    return triggered_actions

def execute_automation_actions(actions):
    """Executes automation actions and logs them for traceability."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    for action in actions:
        print(f"⚡ Executing Automation Action: {action}")
        
        # ✅ Log action execution to iot_automation_logs table
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO iot_automation_logs (timestamp, action) VALUES (?, ?)", (timestamp, action))

        # ✅ Simulate real-world action execution
        if "Send Alert" in action:
            print("📩 Sending alert notification...")  # Replace with real email/SMS function
        elif "Shut Down System" in action:
            print("🚨 Emergency shutdown triggered!")  # Replace with real API call
        elif "Adjust Temperature" in action:
            print("❄️ Adjusting temperature control...")  # Replace with IoT control API

    conn.commit()
    conn.close()
    
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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai  # Ensure OpenAI is installed and configured

# ✅ AI API Key (Ensure it's set up)
openai.api_key = "your-openai-api-key"  # Replace with your actual API key

app = FastAPI()

# ✅ Define the Request Model
class VoiceQuestionRequest(BaseModel):
    question: str

# ✅ API Endpoint to Process the Voice Question
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import sqlite3
import datetime
import os

app = FastAPI()

# ✅ Set up OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment

# ✅ Define the Request Model
class VoiceQuestionRequest(BaseModel):
    question: str

@app.post("/voice-ask")
def process_voice_question(request: VoiceQuestionRequest):
    """Processes voice-input business questions and returns AI-generated insights."""

    question_text = request.question.strip()
    
    if not question_text:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ✅ Fetch the AI model selection from the database (or use a default model)
    conn = sqlite3.connect("ai_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT model FROM ai_model_selection ORDER BY id DESC LIMIT 1")
    selected_model = cursor.fetchone()
    conn.close()

    ai_model = selected_model[0] if selected_model else "text-davinci-003"  # Default model

    try:
        # ✅ Generate AI response
        response = openai.Completion.create(
            engine=ai_model,
            prompt=f"Business AI, answer this question: {question_text}",
            max_tokens=200
        )
        ai_answer = response["choices"][0]["text"].strip()

        # ✅ Store the question and AI response in the database
        conn = sqlite3.connect("ai_data.db")
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO voice_questions (timestamp, question, response) VALUES (?, ?, ?)",
            (timestamp, question_text, ai_answer)
        )
        conn.commit()
        conn.close()

        return {"answer": ai_answer}

    except Exception as e:
        print(f"❌ Error processing voice question: {e}")
        raise HTTPException(status_code=500, detail="Error processing AI response.")

@app.get("/voice-ask-history")
def fetch_voice_question_history():
    """Fetches all stored voice questions and responses."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, question, response FROM voice_questions ORDER BY timestamp DESC LIMIT 50")
    history = [{"timestamp": row[0], "question": row[1], "response": row[2]} for row in cursor.fetchall()]
    conn.close()
    return {"history": history}
    
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
    """Fetches the latest IoT sensor data, detects anomalies, and executes automation actions."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, sensor, value FROM iot_sensors ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    latest_data = {"timestamp": row[0], "sensor": row[1], "value": row[2]}

    # ✅ Detect anomalies
    anomalies = detect_anomalies(row[1])
    if anomalies:
        latest_data["anomaly_detected"] = True
        latest_data["alert_message"] = f"🚨 Anomaly detected in {row[1]}: {row[2]}!"
    else:
        latest_data["anomaly_detected"] = False

    # ✅ Check automation rules
    triggered_actions = check_automation_rules(latest_data)
    if triggered_actions:
        latest_data["triggered_actions"] = triggered_actions
        execute_automation_actions(triggered_actions)  # ✅ NEW: Executes actions

    return latest_data
    
    # Detect anomalies in real-time
    anomalies = detect_anomalies(row[1])

    # If anomaly detected, return it as an alert
    if anomalies:
        latest_data["anomaly_detected"] = True
        latest_data["alert_message"] = f"🚨 Anomaly detected in {row[1]}: {row[2]}!"
    else:
        latest_data["anomaly_detected"] = False

    return latest_data
    
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

            # ✅ Check if any automation rule matches this data
            triggered_actions = check_automation_rules(latest_iot_data)
            if triggered_actions:
                latest_iot_data["triggered_actions"] = triggered_actions

            await websocket.send_json(latest_iot_data)
            await asyncio.sleep(5)  # Send updates every 5 seconds
    except WebSocketDisconnect:
        print("❌ IoT WebSocket Disconnected")
        
@app.get("/fetch-anomalies")
def fetch_anomalies():
    """Fetch all detected anomalies."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, sensor, value, anomaly_score, status FROM iot_anomalies ORDER BY timestamp DESC LIMIT 50")
    anomalies = [{"timestamp": row[0], "sensor": row[1], "value": row[2], "anomaly_score": row[3], "status": row[4]} for row in cursor.fetchall()]
    conn.close()
    return {"anomalies": anomalies}
# ---------------------------------------
# 🚀 AI-Powered Predictive Analytics
# ---------------------------------------
@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Optimized AI Model Selection based on user choice."""

    today = datetime.date.today()
    past_days = 60
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(past_days)][::-1]
    y = np.array([random.uniform(5000, 20000) for _ in range(past_days)])

    selected_model = request.model.lower()

    if selected_model == "linear_regression":
        model = LinearRegression()
        model.fit(np.arange(past_days).reshape(-1, 1), y)
        future_predictions = model.predict(np.arange(past_days, past_days + request.future_days).reshape(-1, 1)).tolist()

    elif selected_model == "arima":
        model = ARIMA(y, order=(5,1,0))
        fitted_model = model.fit()
        future_predictions = fitted_model.forecast(steps=request.future_days).tolist()

    elif selected_model == "prophet":
        df = pd.DataFrame({"ds": dates, "y": y})
        prophet_model = Prophet()
        prophet_model.fit(df)
        future_df = pd.DataFrame({"ds": [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.future_days)]})
        forecast = prophet_model.predict(future_df)
        future_predictions = forecast["yhat"].tolist()

    elif selected_model == "isolation_forest":
        # Isolation Forest does not predict future values, but detects anomalies in the dataset
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(y.reshape(-1, 1))
        anomaly_scores = model.decision_function(y.reshape(-1, 1)).tolist()
        future_predictions = anomaly_scores[-request.future_days:]  # Return recent anomaly scores

    else:
        # If "auto" is selected or invalid input, choose the best model dynamically
        if request.future_days <= 7:
            model = LinearRegression()
            model.fit(np.arange(past_days).reshape(-1, 1), y)
            future_predictions = model.predict(np.arange(past_days, past_days + request.future_days).reshape(-1, 1)).tolist()
        elif request.future_days <= 30:
            model = ARIMA(y, order=(5,1,0))
            fitted_model = model.fit()
            future_predictions = fitted_model.forecast(steps=request.future_days).tolist()
        else:
            df = pd.DataFrame({"ds": dates, "y": y})
            prophet_model = Prophet()
            prophet_model.fit(df)
            future_df = pd.DataFrame({"ds": [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.future_days)]})
            forecast = prophet_model.predict(future_df)
            future_predictions = forecast["yhat"].tolist()

    return {
        "category": request.category,
        "model_used": selected_model if selected_model != "auto" else "auto-selected",
        "predicted_values": future_predictions
    }

@app.post("/get-recommendations")
def get_recommendations(request: RecommendationRequest):
    """Generate AI-based business recommendations based on predictions."""
    category = request.category
    predicted_values = request.predicted_values

    # ✅ Basic recommendation logic (example)
    avg_future_value = sum(predicted_values) / len(predicted_values)
    
    recommendations = []
    if avg_future_value > 10000:
        recommendations.append(f"💡 Consider investing more in {category} due to high projected growth.")
    else:
        recommendations.append(f"⚠️ Be cautious with {category}, as projections show slow growth.")

    return {"recommendations": recommendations}
    

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
import asyncio

async def update_ai_thresholds():
    """Periodically updates AI-adjusted IoT thresholds."""
    while True:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT sensor FROM iot_sensors")
        sensors = [row[0] for row in cursor.fetchall()]
        conn.close()

        for sensor in sensors:
            adjust_thresholds(sensor)

        print("✅ AI-adjusted thresholds updated for all sensors.")
        await asyncio.sleep(3600)  # Update every 1 hour

@app.on_event("startup")
async def start_threshold_updater():
    """Runs the AI threshold adjustment in the background."""
    asyncio.create_task(update_ai_thresholds())
    
# ---------------------------------------
# 🚀 Run the App
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
