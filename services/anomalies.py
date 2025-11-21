# services/anomalies.py

import sqlite3
import numpy as np
import datetime
from sklearn.ensemble import IsolationForest
import openai  # Make sure openai is installed and your API key is configured

DATABASE = "ai_data.db"


def detect_anomalies(sensor_name):
    """Detect anomalies in historical sensor data using Isolation Forest."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT value FROM iot_sensors
        WHERE sensor = ?
        ORDER BY timestamp DESC LIMIT 100
    """, (sensor_name,))
    
    values = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(values) < 10:
        return []

    data = np.array(values).reshape(-1, 1)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data)

    predictions = model.predict(data)
    anomaly_scores = model.decision_function(data)

    # Capture anomalies (label -1)
    anomalies = [values[i] for i in range(len(predictions)) if predictions[i] == -1]

    # Optionally log anomaly to database
    if anomalies:
        log_anomalies(sensor_name, values, anomaly_scores, predictions)
        for anomaly_value in anomalies:
            generate_ai_anomaly_explanation(sensor_name, anomaly_value)

    return anomalies


def log_anomalies(sensor, values, scores, labels):
    """Log the latest anomalies to the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    for i, label in enumerate(labels):
        if label == -1:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO iot_anomalies (timestamp, sensor, value, anomaly_score, status)
                VALUES (?, ?, ?, ?, ?)
            """, (ts, sensor, values[i], scores[i], "ALERT"))

    conn.commit()
    conn.close()


def generate_ai_anomaly_explanation(sensor, anomaly_details):
    """Use OpenAI to explain the anomaly, suggest fixes, and store them."""
    explanation_prompt = f"""
    A sensor named {sensor} reported an anomaly with the following value: {anomaly_details}.
    Explain the possible cause, suggest a fix, and recommend how to prevent it in the future.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": explanation_prompt}]
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        result = f"Error generating explanation: {str(e)}"

    # Save to DB
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ai_anomaly_explanations (sensor, anomaly_details, explanation, timestamp)
        VALUES (?, ?, ?, ?)
    """, (sensor, str(anomaly_details), result, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
