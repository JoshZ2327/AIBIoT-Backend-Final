from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from services.customer_behavior_forecasting import CustomerBehaviorForecastingModel

router = APIRouter()

class ForecastRequest(BaseModel):
    data: list  # list of dicts representing rows of customer behavior data
    target: str

@router.post("/forecast-customer-behavior")
def forecast_behavior(request: ForecastRequest):
    try:
        df = pd.DataFrame(request.data)
        model = CustomerBehaviorForecastingModel(df.copy(), request.target)
        model.preprocess_data()
        model.train_model()
        mae = model.evaluate_model()
        return {"mae": mae}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
