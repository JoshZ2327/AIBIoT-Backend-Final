# routers/network_degradation.py

from fastapi import APIRouter
from pydantic import BaseModel
from services.network_degradation_model import predict_degradation

router = APIRouter()

class NetworkData(BaseModel):
    latency: float
    packet_loss: float
    jitter: float
    throughput: float

@router.post("/predict-network-degradation")
def predict(data: NetworkData):
    result = predict_degradation(data.dict())
    return {
        "input": data.dict(),
        "degradation_detected": result["degradation_predicted"],
        "confidence_score": f"{result['confidence']}%"
    }
