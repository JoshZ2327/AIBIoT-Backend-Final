import os
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import openai

# Set your OpenAI API key from an environment variable.
openai.api_key = os.environ.get("OPENAI_API_KEY")

app = FastAPI(title="AIBIoT AI Engine Backend")

# Enable CORS for development (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo purposes.
users_db = {}           # { email: User }
data_store = {}         # { source: [data, ...] }
training_status_db = {} # { email: status }

# --- Data Models ---
class User(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None

class DataIngestion(BaseModel):
    source: str
    data: dict

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

@app.post("/signup")
async def signup(user: User):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    users_db[user.email] = user
    training_status_db[user.email] = "not started"
    return {"message": "User registered successfully"}

@app.post("/login")
async def login(user: User):
    if user.email not in users_db or users_db[user.email].password != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful"}

@app.post("/query", response_model=QueryResponse)
async def query_data(query_request: QueryRequest):
    try:
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=query_request.query,
            max_tokens=150,
            temperature=0.7,
        )
        response_text = completion.choices[0].text.strip()
    except Exception as e:
        response_text = f"Error generating AI response: {e}"
    return QueryResponse(response=response_text)

def simulate_training(email: str):
    time.sleep(5)
    training_status_db[email] = "completed"

@app.post("/train")
async def train_model(user: User, background_tasks: BackgroundTasks):
    if user.email not in users_db:
        raise HTTPException(status_code=400, detail="User not found; please sign up first.")
    training_status_db[user.email] = "in progress"
    background_tasks.add_task(simulate_training, user.email)
    return {"message": "Model training started", "training_status": training_status_db[user.email]}

@app.get("/status/{email}")
async def get_training_status(email: str):
    status = training_status_db.get(email, "not started")
    return {"email": email, "training_status": status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
