# services/data_pipeline.py

import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np

DATABASE = "aibiot.db"

def ingest_sensor_data(sensor_name: str, value: float):
    """Inserts new sensor data into the database."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO iot_sensors (timestamp, sensor, value) VALUES (?, ?, ?)",
        (timestamp, sensor_name, value)
    )
    conn.commit()
    conn.close()
    return {"status": "success", "timestamp": timestamp}


def fetch_transformed_data(sensor_name: str, days: int = 7):
    """Fetches and transforms sensor data for modeling."""
    conn = sqlite3.connect(DATABASE)
    query = f"""
        SELECT timestamp, value FROM iot_sensors
        WHERE sensor = ? ORDER BY timestamp DESC LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(sensor_name, days * 24 * 60))  # up to 1 reading/min
    conn.close()

    if df.empty:
        return pd.DataFrame()

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Transformation: rolling average & normalization
    df["rolling_avg"] = df["value"].rolling(window=5, min_periods=1).mean()
    df["normalized"] = (df["value"] - df["value"].mean()) / df["value"].std()

    return df
