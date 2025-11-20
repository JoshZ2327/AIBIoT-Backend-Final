from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

router = APIRouter()

class PredictRequest(BaseModel):
    input_data: list

@router.post("/predict")
async def predict(request: PredictRequest):
    model_path = os.path.join("models", "model.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model not found")

    model = joblib.load(model_path)

    try:
        input_array = np.array(request.input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
