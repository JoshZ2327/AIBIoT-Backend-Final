# routers/websocket.py

import asyncio
import sqlite3
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.alerts import check_alerts
from core.anomaly import detect_anomalies
from services.alerts import websocket_alerts, notify_alert_clients
from services.automation import (
    check_automation_rules,
    execute_automation_actions,
)

DATABASE = "aibiot.db"
router = APIRouter()

# --- DATA SOURCES WEBSOCKET ---
@router.websocket("/ws/data-sources")
async def websocket_data_sources(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = {
                "type": "update",
                "data_sources": fetch_data_sources_from_db()
            }
            await websocket.send_json(data)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        print("❌ Data Sources WebSocket Disconnected")

# --- IOT WEBSOCKET ---
@router.websocket("/ws/iot")
async def websocket_iot(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            latest_iot_data = fetch_latest_iot_data()
            await websocket.send_json(latest_iot_data)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        print("❌ IoT WebSocket Disconnected")

# --- HELPERS ---
def fetch_data_sources_from_db():
    """Fetch the latest list of data sources from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, type, path FROM data_sources")
    sources = [{"name": row[0], "type": row[1], "path": row[2]} for row in cursor.fetchall()]
    conn.close()
    return sources

def fetch_latest_iot_data():
    """Fetch latest IoT data, detect anomalies, and execute automation."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, sensor, value FROM iot_sensors ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    latest_data = {"timestamp": row[0], "sensor": row[1], "value": row[2]}
    anomalies = detect_anomalies(row[1])
    if anomalies:
        latest_data["anomaly_detected"] = True
        latest_data["alert_message"] = f"🚨 Anomaly detected in {row[1]}: {row[2]}!"
    else:
        latest_data["anomaly_detected"] = False

    
    return latest_data
