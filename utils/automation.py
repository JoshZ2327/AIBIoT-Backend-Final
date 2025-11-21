# utils/automation.py

def check_automation_rules(sensor_data: dict):
    triggered = []
    if sensor_data["sensor"] == "temperature" and sensor_data["value"] > 75:
        triggered.append("Turn on cooling system")
    if sensor_data.get("anomaly_detected"):
        triggered.append("Send technician alert")
    return triggered

def execute_automation_actions(actions: list):
    for action in actions:
        print(f"✅ Executing action: {action}")
