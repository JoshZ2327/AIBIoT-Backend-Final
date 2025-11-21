from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your routers
from routers.voice import router as voice_router
from routers.prediction import router as prediction_router
from routers.websocket_routes import router as websocket_router
from routers.ingestion import router as ingestion_router

# Create the FastAPI app
app = FastAPI(
    title="AIBIoT Platform",
    description="AI-Driven IoT Monitoring and Automation",
    version="1.0.0"
)

# Allow all CORS (for testing; tighten this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(voice_router, prefix="/voice", tags=["Voice Commands"])
app.include_router(prediction_router, prefix="/predict", tags=["AI Prediction"])
app.include_router(websocket_router, tags=["WebSockets"])
app.include_router(ingestion_router, prefix="/ingestion", tags=["Data Ingestion"])
