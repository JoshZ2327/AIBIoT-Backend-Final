# routers/ingestion.py

from fastapi import APIRouter
from pydantic import BaseModel
from services.data_pipeline import ingest_sensor_data

router = APIRouter()

class SensorData(BaseModel):
    sensor_name: str
    value: float

@router.post("/ingest-sensor")
def ingest_sensor(data: SensorData):
    return ingest_sensor_data(data.sensor_name, data.value)
