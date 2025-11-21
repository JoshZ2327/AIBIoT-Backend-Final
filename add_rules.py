import sqlite3

DATABASE = "ai_data.db"

rules = [
    {
        "sensor_name": "temperature",
        "condition": ">",
        "threshold": 75.0,
        "action": "Turn on fan"
    },
    {
        "sensor_name": "humidity",
        "condition": "<",
        "threshold": 30.0,
        "action": "Activate humidifier"
    },
    {
        "sensor_name": "pressure",
        "condition": ">",
        "threshold": 1015.0,
        "action": "Open pressure valve"
    }
]

def insert_rules():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    for rule in rules:
        cursor.execute("""
            INSERT INTO automation_rules (sensor_name, condition, threshold, action)
            VALUES (?, ?, ?, ?)
        """, (rule["sensor_name"], rule["condition"], rule["threshold"], rule["action"]))

    conn.commit()
    conn.close()
    print("✅ Sample automation rules inserted.")

if __name__ == "__main__":
    insert_rules()
