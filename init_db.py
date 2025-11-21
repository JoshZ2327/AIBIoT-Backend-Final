import sqlite3

DATABASE = "ai_data.db"

def initialize_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Table for sensor data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS iot_sensors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor TEXT,
            value REAL
        )
    """)

    # Table for detected anomalies
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS iot_anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor TEXT,
            value REAL,
            anomaly_score REAL,
            status TEXT
        )
    """)

    # Table for automation rules
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS automation_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor TEXT,
            condition TEXT,
            value REAL,
            action TEXT
        )
    """)

    # Table for alert messages
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            message TEXT
        )
    """)

    # Table for connected data sources
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT,
            path TEXT
        )
    """)

    # Table for AI-generated anomaly explanations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_anomaly_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor TEXT,
            anomaly_details TEXT,
            explanation TEXT,
            timestamp TEXT
        )
    """)

    # Create table for automation rules
cursor.execute("""
    CREATE TABLE IF NOT EXISTS automation_rules (
        rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sensor_name TEXT NOT NULL,
        condition TEXT NOT NULL,  -- e.g., '>', '<', '=='
        threshold REAL NOT NULL,
        action TEXT NOT NULL
    )
""")

# Create table for automation logs
cursor.execute("""
    CREATE TABLE IF NOT EXISTS automation_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        sensor TEXT NOT NULL,
        action TEXT NOT NULL,
        value REAL NOT NULL
    )
""")

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")

if __name__ == "__main__":
    initialize_database()
