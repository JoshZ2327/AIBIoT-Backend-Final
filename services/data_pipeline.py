import sqlite3
from utils.anomaly_detection import detect_anomalies
from utils.automation import check_automation_rules, execute_automation_actions
from datetime import datetime

DATABASE = "aibiot.db"

def save_to_db(sensor_name: str, value: float):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    cursor.execute("INSERT INTO iot_sensors (timestamp, sensor, value) VALUES (?, ?, ?)", (timestamp, sensor_name, value))
    conn.commit()
    conn.close()

def ingest_sensor_data(sensor_name: str, value: float):
    # Save raw data
    save_to_db(sensor_name, value)

    result = {
        "sensor": sensor_name,
        "value": value,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Detect anomalies
    if detect_anomalies(sensor_name):
        result["anomaly_detected"] = True
        result["alert"] = f"Anomaly detected in {sensor_name}!"
    else:
        result["anomaly_detected"] = False

    # Run automation rules
    triggered = check_automation_rules(result)
    if triggered:
        result["triggered_actions"] = triggered
        execute_automation_actions(triggered)

    return result
