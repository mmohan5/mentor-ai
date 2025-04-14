from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from uuid import uuid4
import asyncio
# import uvicorn

from BusinessChatbotEngine import BusinessPlanBuilder, BusinessPlanState

app = FastAPI()
engine = BusinessPlanBuilder()

# In-memory session store
sessions: Dict[str, BusinessPlanState] = {}

class UserInput(BaseModel):
    session_id: str
    user_input: str

@app.post("/start")
async def start():
    session_id = str(uuid4())
    engine = BusinessPlanBuilder()

    asyncio.create_task(engine.invoke())
    sessions[session_id] = engine

    return {
        "session_id": session_id,
        "output": engine.output,
        "allow_input": engine.allow_input
    }

@app.post("/step")
async def step(user_input: UserInput):
    session_id = user_input.session_id
    user_text = user_input.user_input

    if session_id not in sessions:
        return {"error": "Invalid session_id"}

    engine = sessions[session_id]
    await engine.set_user_input(user_text)

    return {
        "output": engine.output,
        "allow_input": engine.allow_input
    }

@app.get("/state/{session_id}")
async def get_state(session_id: str):
    if session_id not in sessions:
        return {"error": "Invalid session_id"}

    engine = sessions[session_id]

    is_new_output = engine.output != engine.last_served_output
    if is_new_output:
        engine.last_served_output = engine.output  # mark as "seen"

    return {
        "output": engine.output,
        "allow_input": engine.allow_input,
        "is_new_output": is_new_output
    }



# uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
