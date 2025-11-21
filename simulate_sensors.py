import requests
import random
import time

URL = "http://127.0.0.1:8000/ingest-sensor"

sensors = ["temperature", "humidity", "pressure"]

def simulate():
    while True:
        for sensor in sensors:
            value = {
                "temperature": random.uniform(65, 85),
                "humidity": random.uniform(20, 50),
                "pressure": random.uniform(1010, 1020)
            }[sensor]

            data = {"sensor_name": sensor, "value": value}
            try:
                response = requests.post(URL, json=data)
                print(f"[{sensor}] Sent {value:.2f} → Status {response.status_code}")
            except Exception as e:
                print("Error:", e)

            time.sleep(2)

if __name__ == "__main__":
    simulate()
