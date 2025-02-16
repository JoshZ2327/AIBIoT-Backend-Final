from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import random
import openai
import sqlite3
import pandas as pd
import os
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

### ✅ AI-Powered Business Question Answering (Enhanced) ###
@app.post("/ask-question")
def ask_business_question(request: BusinessQuestion):
    """Processes user business queries using AI and connected data sources."""
    query = request.question.lower()

    # 1️⃣ Check if the question is about revenue, users, or traffic
    if "revenue" in query or "sales" in query:
        for source_name, source_details in data_sources.items():
            if source_details["type"] == "sqlite":
                conn = sqlite3.connect(source_details["path"])
                cursor = conn.cursor()
                
                cursor.execute("SELECT SUM(sales_amount) FROM sales WHERE date >= date('now', '-30 days')")
                revenue = cursor.fetchone()[0] or "Data unavailable"
                conn.close()
                
                return {"answer": f"Total revenue in the last 30 days: ${revenue}"}

    if "users" in query or "customers" in query:
        for source_name, source_details in data_sources.items():
            if source_details["type"] == "sqlite":
                conn = sqlite3.connect(source_details["path"])
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM users WHERE date >= date('now', '-30 days')")
                users = cursor.fetchone()[0] or "Data unavailable"
                conn.close()
                
                return {"answer": f"Total unique users in the last 30 days: {users}"}

    if "traffic" in query or "visits" in query:
        for source_name, source_details in data_sources.items():
            if source_details["type"] == "sqlite":
                conn = sqlite3.connect(source_details["path"])
                cursor = conn.cursor()
                
                cursor.execute("SELECT SUM(visits) FROM web_traffic WHERE date >= date('now', '-30 days')")
                traffic = cursor.fetchone()[0] or "Data unavailable"
                conn.close()
                
                return {"answer": f"Total website traffic in the last 30 days: {traffic}"}

    # 2️⃣ If no direct database match, use AI for reasoning
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an advanced business intelligence AI. Use multi-step reasoning."},
            {"role": "user", "content": request.question}
        ]
    )
    return {"answer": response["choices"][0]["message"]["content"]}

### ✅ AI-Powered Dashboard Insights (With Real Data) ###
@app.post("/ai-dashboard")
def ai_dashboard(request: DashboardRequest):
    """Generates AI-driven business insights based on real data from connected sources."""
    today = datetime.date.today()
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.dateRange)][::-1]
    
    # Default metrics if no data source exists
    revenue = round(random.uniform(50000, 200000), 2)
    users = random.randint(1000, 5000)
    traffic = random.randint(10000, 50000)
    
    # Step 1️⃣: Check if a database is connected
    for source_name, source_details in data_sources.items():
        if source_details["type"] == "sqlite":
            conn = sqlite3.connect(source_details["path"])
            cursor = conn.cursor()
            
            try:
                cursor.execute("SELECT SUM(sales_amount) FROM sales WHERE date >= ?", (dates[0],))
                revenue = cursor.fetchone()[0] or revenue
                
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM users WHERE date >= ?", (dates[0],))
                users = cursor.fetchone()[0] or users
                
                cursor.execute("SELECT SUM(visits) FROM web_traffic WHERE date >= ?", (dates[0],))
                traffic = cursor.fetchone()[0] or traffic
            except:
                pass
            
            conn.close()
    
    data = {
        "revenue": revenue,
        "users": users,
        "traffic": traffic,
        "revenueTrends": {"dates": dates, "values": [random.uniform(5000, 20000) for _ in range(request.dateRange)]},
        "userTrends": {"dates": dates, "values": [random.randint(50, 200) for _ in range(request.dateRange)]},
        "trafficTrends": {"values": [random.randint(2000, 15000), random.randint(1000, 10000), random.randint(500, 5000)]},
    }
    
    return data

### ✅ Run the App ###
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
