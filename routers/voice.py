from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Define the request body structure
class VoiceCommand(BaseModel):
    command: str

# Simulated voice command handler
@router.post("/voice-command")
async def handle_voice_command(command: VoiceCommand):
    # Example logic to interpret the voice command
    if command.command.lower() == "turn on lights":
        return {"response": "Lights turned on."}
    elif command.command.lower() == "turn off lights":
        return {"response": "Lights turned off."}
    else:
        raise HTTPException(status_code=400, detail="Unknown voice command.")
