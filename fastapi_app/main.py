
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import json
from datetime import datetime
import os


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from langgraph_stance_analyzer.main import app as langgraph_app

app = FastAPI()

origins = [
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AGENT_RUNS_DIR = "/home/rgukt/Documents/major project/major-project/agent_runs"
os.makedirs(AGENT_RUNS_DIR, exist_ok=True)

class RunAgentRequest(BaseModel):
    text: str

class AgentRunResponse(BaseModel):
    run_id: str
    status: str
    input_text: str
    result: dict | None = None
    timestamp: datetime

@app.post("/run_agent", response_model=AgentRunResponse)
async def run_agent(request: RunAgentRequest):
    run_id = str(uuid.uuid4())
    timestamp = datetime.now()
    input_text = request.text

    try:
        initial_state = {"input": input_text, "target": "", "max_turns": 3}
        result = langgraph_app.invoke(initial_state)
        status = "completed"
    except Exception as e:
        result = {"error": str(e)}
        status = "failed"

    run_data = {
        "run_id": run_id,
        "status": status,
        "input_text": input_text,
        "result": result,
        "timestamp": timestamp.isoformat()
    }

    file_path = os.path.join(AGENT_RUNS_DIR, f"{run_id}.json")
    with open(file_path, "w") as f:
        json.dump(run_data, f, indent=4)

    return AgentRunResponse(run_id=run_id, status=status, input_text=input_text, result=result, timestamp=timestamp)

@app.get("/agent_runs", response_model=list[AgentRunResponse])
async def get_all_agent_runs():
    runs = []
    for filename in os.listdir(AGENT_RUNS_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(AGENT_RUNS_DIR, filename)
            with open(file_path, "r") as f:
                run_data = json.load(f)
                runs.append(AgentRunResponse(**run_data))
    return runs

@app.get("/agent_runs/{run_id}", response_model=AgentRunResponse)
async def get_agent_run(run_id: str):
    file_path = os.path.join(AGENT_RUNS_DIR, f"{run_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Agent run not found")
    
    with open(file_path, "r") as f:
        run_data = json.load(f)
    return AgentRunResponse(**run_data)
