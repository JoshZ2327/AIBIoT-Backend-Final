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
    type: str  # 'sqlite', 'csv', 'api'
    path: str  # File path, database URI, or API URL

class BusinessQuestion(BaseModel):
    question: str

class DashboardRequest(BaseModel):
    dateRange: int
    category: str

class PredictionRequest(BaseModel):
    category: str  # 'revenue', 'users', 'traffic'
    future_days: int  # Forecast period (1-90 days)

class AnomalyDetectionRequest(BaseModel):
    category: str  # 'revenue', 'users', 'traffic'
    values: list  # Historical values for anomaly detection

class RecommendationRequest(BaseModel):
    category: str
    predicted_values: list

@app.get("/")
def read_root():
    return {"message": "Welcome to AIBIoT Backend API!"}

### ✅ IoT Data Monitoring ###
@app.get("/latest-iot-data")
def get_latest_iot_data():
    return {"latest_reading": iot_data[-1]}

@app.get("/iot-history")
def get_iot_history():
    return {"history": iot_data}

### ✅ AI-Powered Predictive Maintenance ###
@app.post("/predict-maintenance")
def predict_maintenance(request: MaintenanceRequest):
    next_maintenance = datetime.datetime.utcnow() + datetime.timedelta(days=random.randint(5, 30))
    return {"next_maintenance": next_maintenance.strftime("%Y-%m-%d")}

### ✅ AI-Powered Spare Parts Inventory ###
@app.get("/check-inventory")
def check_inventory():
    restock_recommendations = []
    for part, details in spare_parts_inventory.items():
        if details["current_stock"] <= details["min_stock"]:
            restock_recommendations.append({
                "part_name": part,
                "current_stock": details["current_stock"],
                "recommended_order_quantity": details["min_stock"] * 2,
                "supplier": details["supplier"]
            })
    return {"restock_recommendations": restock_recommendations}

@app.post("/generate-restock-order")
def generate_restock_order(request: RestockOrderRequest):
    if request.part_name not in spare_parts_inventory:
        raise HTTPException(status_code=404, detail="Part not found")

    supplier = spare_parts_inventory[request.part_name]["supplier"]
    return {"message": f"Restock order placed for {request.part_name} from {supplier}"}

@app.post("/schedule-maintenance")
def schedule_maintenance(request: MaintenanceRequest):
    technician = random.choice(["Technician A", "Technician B", "Technician C"])
    return {"technician": technician}

### ✅ Connect Data Sources ###
@app.post("/connect-data-source")
def connect_data_source(request: DataSourceRequest):
    """Allows users to connect SQL databases, CSV files, or APIs."""
    if request.type not in ["sqlite", "csv", "api"]:
        raise HTTPException(status_code=400, detail="Unsupported data source type")
    
    data_sources[request.name] = {
        "type": request.type,
        "path": request.path
    }
    return {"message": f"Data source '{request.name}' connected successfully"}

### ✅ AI-Powered Predictive Analytics ###
@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Predicts future trends for revenue, users, or traffic using AI models."""
    
    today = datetime.date.today()
    past_days = 30
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(past_days)][::-1]

    historical_data = [(dates[i], random.uniform(5000, 20000)) for i in range(past_days)]

    # Step 1️⃣: Prepare Data for Prediction
    X = np.array([i for i in range(len(historical_data))]).reshape(-1, 1)
    y = np.array([val[1] for val in historical_data])

    # Step 2️⃣: Train AI Model (Linear Regression)
    model = LinearRegression()
    model.fit(X, y)

    # Step 3️⃣: Predict Future Values
    future_X = np.array([len(historical_data) + i for i in range(request.future_days)]).reshape(-1, 1)
    future_predictions = model.predict(future_X).tolist()
    
    future_dates = [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.future_days)]

    return {
        "category": request.category,
        "predicted_trends": {
            "dates": future_dates,
            "values": future_predictions
        }
    }

### ✅ AI-Powered Anomaly Detection ###
@app.post("/detect-anomalies")
def detect_anomalies(request: AnomalyDetectionRequest):
    """Detects anomalies in business data trends using an Isolation Forest model."""

    values = np.array(request.values).reshape(-1, 1)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(values)

    anomaly_scores = model.decision_function(values).tolist()
    anomaly_labels = model.predict(values).tolist()

    anomalies = [{"value": request.values[i], "score": anomaly_scores[i], "is_anomaly": anomaly_labels[i] == -1} 
                 for i in range(len(request.values))]

    return {"category": request.category, "anomalies": anomalies}

### ✅ AI-Powered Business Recommendations ###
@app.post("/ai-recommendations")
def ai_recommendations(request: RecommendationRequest):
    """Generates AI-driven recommendations based on predicted trends."""

    recommendations = []

    if request.category == "revenue":
        avg_revenue = sum(request.predicted_values) / len(request.predicted_values)
        if avg_revenue < 50000:
            recommendations.append("🚀 Consider running a marketing campaign to boost sales.")
        elif avg_revenue > 150000:
            recommendations.append("📈 Invest in scaling operations for future growth.")

    if request.category == "users":
        avg_users = sum(request.predicted_values) / len(request.predicted_values)
        if avg_users < 1000:
            recommendations.append("📢 Improve onboarding experience to retain users.")
        elif avg_users > 3000:
            recommendations.append("🔄 Increase customer support to handle higher demand.")

    if request.category == "traffic":
        avg_traffic = sum(request.predicted_values) / len(request.predicted_values)
        if avg_traffic < 10000:
            recommendations.append("📌 Optimize SEO and run social media ads.")
        elif avg_traffic > 30000:
            recommendations.append("📊 Analyze visitor behavior to improve conversions.")

    return {"category": request.category, "recommendations": recommendations}

### ✅ Run the App ###
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
