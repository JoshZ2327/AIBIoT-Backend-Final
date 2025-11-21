# services/alerts.py

import sqlite3
from fastapi.websockets import WebSocket
from starlette.websockets import WebSocketDisconnect

from services.anomalies import detect_anomalies
from services.automation import check_automation_rules, execute_automation_actions
from services.database import get_db_connection

DATABASE = "iot_data.db"
alert_connections = []

async def websocket_alerts(websocket: WebSocket):
    """WebSocket connection for real-time alerts."""
    await websocket.accept()
    alert_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        alert_connections.remove(websocket)

async def notify_alert_clients():
    """Send real-time AI alerts via WebSocket."""
    alerts = check_alerts()
    for connection in alert_connections:
        try:
            await connection.send_json(alerts)
        except:
            alert_connections.remove(connection)

def check_alerts():
    """Placeholder function for generating current alerts."""
    # You can expand this logic to run anomaly checks across all sensors
    return [{"type": "alert", "message": "🚨 Sample alert message"}]
