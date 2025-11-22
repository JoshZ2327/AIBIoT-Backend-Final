# routers/pnsdm_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from services.pnsdm import PredictiveNetworkServiceDegradationModel

router = APIRouter()

class PNSDMRequest(BaseModel):
    sensor_name: str
    sensor_data: list[float]
    future_arima_days: int = 7
    future_prophet_days: int = 30

@router.post("/analyze-pnsdm")
def run_pnsdm(request: PNSDMRequest):
    model = PredictiveNetworkServiceDegradationModel(
        sensor_data=request.sensor_data,
        sensor_name=request.sensor_name
    )
    model.forecast_trend_arima(future_steps=request.future_arima_days)
    model.forecast_trend_prophet(future_steps=request.future_prophet_days)
    model.detect_anomalies()
    result = model.results
    return {"analysis": result}
