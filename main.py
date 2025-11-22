from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import voice, prediction, websocket, ingestion
from database.init import init_db
from services.storage import move_old_data_to_cold_storage
from services.anomalies import update_ai_thresholds
import asyncio
import os
import openai
from routers import digital_twin

app.include_router(digital_twin.router)
from routers import network_degradation

app.include_router(network_degradation.router)
from routers import pnsdm_router
app.include_router(pnsdm_router.router)

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Register Routers
app.include_router(voice.router)
app.include_router(prediction.router)
app.include_router(websocket.router)
app.include_router(ingestion.router)

# ✅ Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Set API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Initialize SQLite DB (creates tables if not exist)
init_db()

# ✅ Background Task: Cold Storage
@app.on_event("startup")
async def schedule_storage_optimization():
    """Periodically move old data to cold storage."""
    while True:
        move_old_data_to_cold_storage()
        await asyncio.sleep(86400)  # Every 24 hours

# ✅ Background Task: Update AI Thresholds
@app.on_event("startup")
async def start_threshold_updater():
    """Continuously updates AI thresholds."""
    asyncio.create_task(update_ai_thresholds())

# ✅ Run locally (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
