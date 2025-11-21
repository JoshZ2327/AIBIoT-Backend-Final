from fastapi import APIRouter

from routers.voice import router as voice_router
from routers.prediction import router as prediction_router
from routers.websocket import router as websocket_router

api_router = APIRouter()

# Register sub-routers
api_router.include_router(voice_router, tags=["Voice Commands"])
api_router.include_router(prediction_router, tags=["AI Predictions"])
api_router.include_router(websocket_router, tags=["WebSockets"])
