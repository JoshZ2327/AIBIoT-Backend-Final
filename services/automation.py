# services/automation.py

import sqlite3
from datetime import datetime

DATABASE = "ai_data.db"

def check_automation_rules(sensor_data):
    """
    Check if any automation rules match the current sensor data.
    Returns a list of triggered actions.
    """
    triggered = []
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT rule_id, sensor_name, condition, threshold, action
        FROM automation_rules
        WHERE sensor_name = ?
    """, (sensor_data["sensor"],))

    rules = cursor.fetchall()
    conn.close()

    for rule in rules:
        rule_id, sensor_name, condition, threshold, action = rule
        value = sensor_data["value"]

        if (
            (condition == ">" and value > threshold) or
            (condition == "<" and value < threshold) or
            (condition == ">=" and value >= threshold) or
            (condition == "<=" and value <= threshold) or
            (condition == "==" and value == threshold)
        ):
            triggered.append({
                "rule_id": rule_id,
                "sensor": sensor_name,
                "action": action,
                "value": value,
                "condition": condition,
                "threshold": threshold
            })

    return triggered


def execute_automation_actions(actions):
    """
    Simulate or execute automation actions.
    This version just logs them in the database.
    """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    for action in actions:
        cursor.execute("""
            INSERT INTO automation_logs (timestamp, sensor, action, value)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            action["sensor"],
            action["action"],
            action["value"]
        ))

    conn.commit()
    conn.close()
