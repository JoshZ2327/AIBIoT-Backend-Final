from fastapi import APIRouter, HTTPException
import sqlite3
import json
import datetime
import numpy as np
from sklearn.ensemble import IsolationForest

router = APIRouter()
DATABASE = "ai_data.db"

@router.post("/add-digital-twin")
def add_digital_twin(asset_name: str, asset_type: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    sensor_data = json.dumps({})
    ai_thresholds = json.dumps({})
    try:
        cursor.execute("INSERT INTO digital_twins (asset_name, asset_type, sensor_data, ai_thresholds) VALUES (?, ?, ?, ?)",
                       (asset_name, asset_type, sensor_data, ai_thresholds))
        conn.commit()
        return {"message": f"✅ Digital Twin '{asset_name}' added successfully!"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail=f"❌ Digital Twin '{asset_name}' already exists.")
    finally:
        conn.close()

@router.get("/get-digital-twins")
def get_digital_twins():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT asset_name, asset_type, sensor_data, ai_thresholds, status FROM digital_twins")
    twins = [{"asset_name": row[0], "asset_type": row[1], "sensor_data": json.loads(row[2]), "ai_thresholds": json.loads(row[3]), "status": row[4]} for row in cursor.fetchall()]
    conn.close()
    return {"digital_twins": twins}

@router.post("/update-digital-twin")
def update_digital_twin(asset_name: str, sensor_data: dict):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    sensor_data_json = json.dumps(sensor_data)
    cursor.execute("UPDATE digital_twins SET sensor_data = ?, last_updated = CURRENT_TIMESTAMP WHERE asset_name = ?",
                   (sensor_data_json, asset_name))
    conn.commit()
    conn.close()
    return {"message": f"✅ Digital Twin '{asset_name}' updated successfully!"}

@router.get("/get-digital-twin-3d")
def get_digital_twin_3d(asset_name: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT asset_name, sensor_data, ai_thresholds FROM digital_twins WHERE asset_name = ?", (asset_name,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"❌ Digital Twin '{asset_name}' not found.")
    
    return {
        "asset_name": row[0],
        "sensor_data": json.loads(row[1]),
        "ai_thresholds": json.loads(row[2])
    }

@router.post("/adjust-ai-thresholds")
def adjust_ai_thresholds(asset_name: str):
    """Adjust thresholds for digital twin using Isolation Forest"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT sensor_data FROM digital_twins WHERE asset_name = ?", (asset_name,))
    row = cursor.fetchone()

    if not row:
        return {"error": f"❌ Digital Twin '{asset_name}' not found."}

    sensor_data = json.loads(row[0])
    values = np.array(list(sensor_data.values())).reshape(-1, 1)

    if len(values) < 10:
        return {"message": f"⚠️ Not enough data to adjust threshold for {asset_name}."}

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(values)
    adjusted_threshold = float(np.percentile(values, 95))
    ai_thresholds = json.dumps({"adjusted_threshold": adjusted_threshold})

    cursor.execute("UPDATE digital_twins SET ai_thresholds = ?, last_updated = CURRENT_TIMESTAMP WHERE asset_name = ?",
                   (ai_thresholds, asset_name))
    conn.commit()
    conn.close()

    return {"message": f"✅ AI thresholds adjusted for {asset_name}."}
