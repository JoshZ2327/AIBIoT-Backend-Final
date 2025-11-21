from fastapi import FastAPI
from routers import voice, prediction, alerts, ingestion
import uvicorn

app = FastAPI(
    title="AIBIoT Platform",
    description="Modular AI-powered IoT automation and analytics platform",
    version="1.0.0"
)

# Include routers
app.include_router(voice.router, prefix="/voice", tags=["Voice Commands"])
app.include_router(prediction.router, prefix="/predict", tags=["Prediction"])
app.include_router(alerts.router, tags=["Alerts"])
app.include_router(ingestion.router, prefix="/sensor", tags=["Sensor Ingestion"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
