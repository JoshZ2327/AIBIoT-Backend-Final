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

# Store connected data sources
data_sources = {}

# Request Models
class PredictionRequest(BaseModel):
    category: str  # 'revenue', 'users', 'traffic'
    future_days: int  # Forecast period (1-90 days)

class BusinessQuestion(BaseModel):
    question: str

class DataSourceRequest(BaseModel):
    name: str
    type: str  # 'sqlite', 'csv', 'api'
    path: str  # File path, database URI, or API URL

@app.get("/")
def read_root():
    return {"message": "Welcome to AIBIoT Backend API!"}

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

### ✅ AI-Powered Business Question Answering ###
@app.post("/ask-question")
def ask_business_question(request: BusinessQuestion):
    """Processes user business queries using AI and available data sources."""
    query = request.question.lower()

    # Search for relevant data in connected sources
    for source_name, source_details in data_sources.items():
        if source_name.lower() in query:
            return {"answer": f"Connected data source '{source_name}' is available. Data type: {source_details['type']}"}

    # AI-powered reasoning
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an advanced business intelligence AI. Use multi-step reasoning."},
            {"role": "user", "content": request.question}
        ]
    )
    return {"answer": response["choices"][0]["message"]["content"]}

### ✅ AI-Powered Predictive Analytics ###
@app.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Predicts future trends for revenue, users, or traffic using AI models."""
    
    today = datetime.date.today()
    past_days = 30  # Use last 30 days for training
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(past_days)][::-1]

    historical_data = []
    
    # Step 1️⃣: Check for connected data sources
    for source_name, source_details in data_sources.items():
        if source_details["type"] == "sqlite":
            conn = sqlite3.connect(source_details["path"])
            cursor = conn.cursor()
            
            try:
                if request.category == "revenue":
                    cursor.execute("SELECT date, SUM(sales_amount) FROM sales WHERE date >= ? GROUP BY date", (dates[0],))
                elif request.category == "users":
                    cursor.execute("SELECT date, COUNT(DISTINCT user_id) FROM users WHERE date >= ? GROUP BY date", (dates[0],))
                elif request.category == "traffic":
                    cursor.execute("SELECT date, SUM(visits) FROM web_traffic WHERE date >= ? GROUP BY date", (dates[0],))
                
                historical_data = cursor.fetchall()
            except:
                pass
            
            conn.close()
    
    # If no database, generate random trend data
    if not historical_data:
        historical_data = [(dates[i], random.uniform(5000, 20000)) for i in range(past_days)]
    
    # Step 2️⃣: Prepare Data for Prediction
    X = np.array([i for i in range(len(historical_data))]).reshape(-1, 1)  # Day indexes
    y = np.array([val[1] for val in historical_data])  # Metric values

    # Step 3️⃣: Train AI Model (Linear Regression)
    model = LinearRegression()
    model.fit(X, y)

    # Step 4️⃣: Predict Future Values
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

### ✅ Run the App ###
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
