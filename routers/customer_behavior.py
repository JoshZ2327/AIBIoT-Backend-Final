# routers/customer_behavior.py

from fastapi import APIRouter
from pydantic import BaseModel
from services.customer_behavior_forecasting import CustomerBehaviorForecastingModel, save_behavior_predictions
import pandas as pd

router = APIRouter()

class ForecastRequest(BaseModel):
    customer_id: str
    data: list  # List of dicts representing input data
    target_column: str

@router.post("/forecast-customer-behavior")
def forecast_customer_behavior(request: ForecastRequest):
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)

        # Train model
        model = CustomerBehaviorForecastingModel(df, request.target_column)
        training_result = model.train()

        # Make prediction on new data (re-use last entry for demo purposes)
        new_input = request.data[-1]
        prediction = model.predict_new([new_input])[0]

        # Log the prediction
        save_behavior_predictions(request.customer_id, new_input, prediction)

        return {
            "rmse": training_result["rmse"],
            "prediction": prediction,
            "feature_importance": training_result["feature_importance"]
        }

    except Exception as e:
        return {"error": str(e)}
