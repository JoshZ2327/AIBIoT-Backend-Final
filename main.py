from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import ingestion, voice, websocket, prediction

app = FastAPI(
    title="AIBIoT Platform",
    description="AI-powered IoT automation and analytics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(ingestion.router)
app.include_router(voice.router)
app.include_router(websocket.router)
app.include_router(prediction.router)

@app.get("/")
def read_root():
    return {"message": "AIBIoT backend is running 🚀"}
