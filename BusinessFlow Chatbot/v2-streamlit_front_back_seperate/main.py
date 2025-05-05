from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from uuid import uuid4
import asyncio
from fastapi.middleware.cors import CORSMiddleware

from BusinessChatbotEngine import BusinessPlanBuilder, BusinessPlanState

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Only allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    
    for _ in range(1800):  # wait 3 min max
        await asyncio.sleep(0.1)
        if engine.output != "":
            break
    
    engine.is_output_ready = False

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
    
    async with engine.input_processed_condition:
        try:
            await asyncio.wait_for(
                engine.input_processed_condition.wait(),
                timeout=3600.0 * 6 # 6 hours
            )
        except asyncio.TimeoutError:
            pass

    if not engine.is_output_ready:
        async with engine.output_ready_condition:
            try:
                await asyncio.wait_for(
                    engine.output_ready_condition.wait(),
                    timeout=180.0 # 3 minutes
                )
            except asyncio.TimeoutError:
                pass

    return {
        "output": engine.output,
        "allow_input": engine.allow_input
    }



# uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
