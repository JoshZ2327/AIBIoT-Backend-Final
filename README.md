AIBIoT Platform – Backend

The AIBIoT platform is a modular, intelligent IoT backend designed for real-time monitoring, anomaly detection, AI prediction, digital twins, voice commands, and rule-based automation. Built with FastAPI, SQLite, and Python-based ML libraries, this backend supports enterprise-grade IoT deployments.

⸻

🚀 Features
	•	Real-time IoT sensor ingestion and analytics
	•	AI-powered anomaly detection (Isolation Forest)
	•	Trend prediction (Linear Regression, ARIMA, Prophet)
	•	Automation rule engine (trigger-based responses)
	•	Voice command interaction (OpenAI powered)
	•	WebSocket-based live alerts and dashboards
	•	Digital twin support for device state visualization
	•	Modular architecture with routers/ and services/ separation

⸻

🏗️ Project Structure

AIBIoT-Backend-Final/
│
├── main.py                  # App entry point (replaces index.py)
├── routers/                 # API endpoints (modularized)
│   ├── ingestion.py
│   ├── prediction.py
│   ├── voice.py
│   └── websocket.py
│
├── services/                # Business logic
│   ├── anomalies.py
│   ├── automation.py
│   ├── data_pipeline.py
│   ├── digital_twins.py
│   └── voice_ai.py
│
├── database/                # DB initialization and helpers
│   └── init_db.py
│
├── requirements.txt         # Python dependencies
└── README.md



⸻

⚙️ Setup Instructions

1. Clone the repository

git clone https://github.com/JoshZ2327/AIBIoT-Backend-Final.git
cd AIBIoT-Backend-Final

2. Install dependencies

pip install -r requirements.txt

3. Create the database schema

python database/init_db.py

4. Run the app

uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for the interactive Swagger UI.

⸻

🔐 Environment Variables

Set the following before running the app:

export OPENAI_API_KEY=your_openai_key
export SENDGRID_API_KEY=your_sendgrid_key



⸻

📦 Deployment

This backend is ready for deployment via:
	•	Docker (recommended)
	•	Gunicorn + Uvicorn
	•	Heroku / AWS / Render / Railway

⸻

🧠 Future Additions
	•	Model training interface
	•	Real-time analytics dashboard
	•	External data integrations

⸻

🧾 License

Proprietary – All rights reserved. Not for public distribution without written consent from Maverick Software.
