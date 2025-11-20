# routers/prediction.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import random

router = APIRouter()

class PredictionRequest(BaseModel):
    category: str
    model: Literal["auto", "linear_regression", "arima", "prophet", "isolation_forest"]
    future_days: int

@router.post("/predict-trends")
def predict_trends(request: PredictionRequest):
    """Optimized AI Model Selection based on user choice."""

    today = datetime.date.today()
    past_days = 60
    dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(past_days)][::-1]
    y = np.array([random.uniform(5000, 20000) for _ in range(past_days)])

    selected_model = request.model.lower()

    if selected_model == "linear_regression":
        model = LinearRegression()
        model.fit(np.arange(past_days).reshape(-1, 1), y)
        future_predictions = model.predict(np.arange(past_days, past_days + request.future_days).reshape(-1, 1)).tolist()

    elif selected_model == "arima":
        model = ARIMA(y, order=(5,1,0))
        fitted_model = model.fit()
        future_predictions = fitted_model.forecast(steps=request.future_days).tolist()

    elif selected_model == "prophet":
        df = pd.DataFrame({"ds": dates, "y": y})
        prophet_model = Prophet()
        prophet_model.fit(df)
        future_df = pd.DataFrame({"ds": [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.future_days)]})
        forecast = prophet_model.predict(future_df)
        future_predictions = forecast["yhat"].tolist()

    elif selected_model == "isolation_forest":
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(y.reshape(-1, 1))
        anomaly_scores = model.decision_function(y.reshape(-1, 1)).tolist()
        future_predictions = anomaly_scores[-request.future_days:]

    else:
        # Auto-selection logic
        if request.future_days <= 7:
            model = LinearRegression()
            model.fit(np.arange(past_days).reshape(-1, 1), y)
            future_predictions = model.predict(np.arange(past_days, past_days + request.future_days).reshape(-1, 1)).tolist()
        elif request.future_days <= 30:
            model = ARIMA(y, order=(5,1,0))
            fitted_model = model.fit()
            future_predictions = fitted_model.forecast(steps=request.future_days).tolist()
        else:
            df = pd.DataFrame({"ds": dates, "y": y})
            prophet_model = Prophet()
            prophet_model.fit(df)
            future_df = pd.DataFrame({"ds": [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(request.future_days)]})
            forecast = prophet_model.predict(future_df)
            future_predictions = forecast["yhat"].tolist()

    return {
        "category": request.category,
        "model_used": selected_model if selected_model != "auto" else "auto-selected",
        "predicted_values": future_predictions
    }
