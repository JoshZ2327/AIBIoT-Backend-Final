import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

class PredictiveNetworkServiceDegradationModel:
    def __init__(self, sensor_data, sensor_name):
        self.sensor_data = sensor_data  # A list of historical values
        self.sensor_name = sensor_name
        self.results = {}

    def detect_anomalies(self):
        """Uses Isolation Forest to detect service degradation anomalies."""
        data = np.array(self.sensor_data).reshape(-1, 1)
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(data)
        predictions = model.predict(data)
        scores = model.decision_function(data)

        anomalies = [self.sensor_data[i] for i in range(len(predictions)) if predictions[i] == -1]

        self.results["anomalies"] = anomalies
        self.results["anomaly_scores"] = scores.tolist()
        return anomalies

    def forecast_trend_arima(self, future_steps=7):
        """Forecasts short-term service metrics using ARIMA."""
        try:
            model = ARIMA(self.sensor_data, order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=future_steps)
            self.results["arima_forecast"] = forecast.tolist()
        except Exception as e:
            self.results["arima_forecast_error"] = str(e)

    def forecast_trend_prophet(self, future_steps=30):
        """Forecasts long-term trends using Prophet."""
        today = datetime.date.today()
        dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(self.sensor_data))][::-1]

        df = pd.DataFrame({
            "ds": pd.to_datetime(dates),
            "y": self.sensor_data
        })

        model = Prophet(daily_seasonality=True)
        model.fit(df)

        future = pd.DataFrame({
            "ds": [today + datetime.timedelta(days=i) for i in range(1, future_steps + 1)]
        })

        forecast = model.predict(future)
        self.results["prophet_forecast"] = forecast[["ds", "yhat"]].to_dict(orient="records")

    def run_model(self):
        """Runs full detection and forecasting."""
        self.detect_anomalies()
        self.forecast_trend_arima()
        self.forecast_trend_prophet()
        return self.results
