from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import random
import openai  # AI-powered question answering
import sqlite3  # Example database integration
import pandas as pd  # CSV support
import os

app = FastAPI()

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

@app.get("/")
def read_root():
    return {"message": "Welcome to AIBIoT Backend API!"}

@app.get("/latest-iot-data")
def get_latest_iot_data():
    latest_data = iot_data[-1]
    return {"latest_reading": latest_data}

@app.get("/iot-history")
def get_iot_history():
    return {"history": iot_data}

@app.post("/predict-maintenance")
def predict_maintenance(request: MaintenanceRequest):
    next_maintenance = datetime.datetime.utcnow() + datetime.timedelta(days=random.randint(5, 30))
    return {"next_maintenance": next_maintenance.strftime("%Y-%m-%d")}

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

### ✅ NEW: Connect Data Sources ###
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

### ✅ NEW: AI-Powered Business Question Answering ###
@app.post("/ask-question")
def ask_business_question(request: BusinessQuestion):
    """Processes user business queries using AI and available data sources."""
    query = request.question.lower()
    
    # If the question is about IoT data
    if "iot" in query or "sensor" in query:
        latest_data = iot_data[-1]
        return {"answer": f"Latest IoT reading: {latest_data['sensor']} - {latest_data['value']} at {latest_data['timestamp']}"}

    # If the question is about spare parts inventory
    if "inventory" in query or "restock" in query:
        inventory_info = check_inventory()
        return {"answer": f"Spare parts that need restocking: {inventory_info}"}

    # If question requires AI-powered reasoning
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a business analyst AI."},
                  {"role": "user", "content": request.question}]
    )
    return {"answer": response["choices"][0]["message"]["content"]}
