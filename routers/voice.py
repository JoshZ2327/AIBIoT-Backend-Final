class VoiceCommandRequest(BaseModel):
    command: str

@app.post("/voice-command")
def process_voice_command(request: VoiceCommandRequest):
    """Processes voice commands and triggers IoT actions."""
    command = request.command.lower()

    # ✅ Define voice-triggered actions
    action_mapping = {
        "turn on the lights": "Turn on Lights",
        "turn off the lights": "Turn off Lights",
        "increase temperature": "Increase Temperature",
        "decrease temperature": "Decrease Temperature",
        "open the door": "Open Door",
        "close the door": "Close Door",
        "activate alarm": "Activate Alarm",
        "deactivate alarm": "Deactivate Alarm"
    }

    matched_action = None
    for phrase, action in action_mapping.items():
        if phrase in command:
            matched_action = action
            break

    if matched_action:
        # ✅ Log action in IoT Automation Logs
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO iot_automation_logs (timestamp, action) VALUES (?, ?)", (timestamp, matched_action))
        conn.commit()
        conn.close()

        return {"message": f"IoT Action Executed: {matched_action}"}

    return {"message": "❌ Command not recognized"}
