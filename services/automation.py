# services/automation.py

import sqlite3
import datetime
import ast

DATABASE = "ai_data.db"

# ---------------------------
# SAFE CONDITION EVALUATION
# ---------------------------

def safe_eval_condition(condition: str, sensor_value: float):
    """
    Safely evaluates automation rule conditions.
    Example: "Temperature > THRESHOLD"
    """
    condition = condition.replace("Temperature", str(sensor_value))

    try:
        node = ast.parse(condition, mode='eval')
        if isinstance(node, ast.Expression):
            return eval(compile(node, "<string>", "eval"))
    except Exception as e:
        print(f"❌ Error evaluating automation rule: {e}")
        return False


# ---------------------------
# AUTOMATION RULE CHECKING
# ---------------------------

def check_automation_rules(iot_data):
    """Checks if any AI-adjusted automation rule matches incoming IoT data."""

    sensor_name = iot_data.get("sensor")
    sensor_value = iot_data.get("value")

    if sensor_name is None:
        return []

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Fetch AI-adjusted threshold
    cursor.execute("SELECT adjusted_threshold FROM ai_adjusted_thresholds WHERE sensor = ?", (sensor_name,))
    entry = cursor.fetchone()
    conn.close()

    if not entry:
        return []  # No threshold available

    threshold_value = entry[0]

    # Fetch rules
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT trigger_condition, action FROM iot_automation_rules")
    rules = cursor.fetchall()
    conn.close()

    triggered_actions = []

    for trigger_condition, action in rules:
        # replace placeholder THRESHOLD with real value
        condition = trigger_condition.replace("THRESHOLD", str(threshold_value))

        try:
            if eval(condition):
                triggered_actions.append(action)
                print(f"🔥 Automation Triggered: {action}")
        except Exception as e:
            print(f"❌ Error evaluating rule '{condition}': {e}")

    return triggered_actions


# ---------------------------
# EXECUTE AUTOMATION ACTIONS
# ---------------------------

def execute_automation_actions(actions):
    """
    Executes automation actions and logs them 
    (in real life, this would call external APIs).
    """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    for action in actions:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            "INSERT INTO iot_automation_logs (timestamp, action) VALUES (?, ?)",
            (ts, action)
        )

        # Simulated device actions
        if "Send Alert" in action:
            print("📩 Sending alert notification...")
        elif "Shut Down System" in action:
            print("🚨 Emergency shutdown triggered!")
        elif "Adjust Temperature" in action:
            print("❄️ Adjusting temperature...")
        else:
            print(f"⚡ Executing: {action}")

    conn.commit()
    conn.close()
