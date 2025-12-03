from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bert_bot import SubstationChatbot
from models.knowledge_base import EQUIPMENT_DATA

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory of the current file (main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Initialize Bot
bot = SubstationChatbot()

class Query(BaseModel):
    question: str

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))

@app.post("/api/chat")
async def chat_endpoint(query: Query):
    response, accuracy = bot.get_response(query.question)
    return {"response": response, "accuracy": accuracy}

@app.get("/api/equipment")
def get_equipment_list():
    return [item["equipment"] for item in EQUIPMENT_DATA]

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# Mock Status API for the dashboard
@app.get("/api/status")
def get_status():
    import random
    voltage = round(random.uniform(215.0, 245.0), 2)
    current = round(random.uniform(50.0, 110.0), 2)
    temperature = round(random.uniform(40.0, 90.0), 2)
    
    prediction = "Normal"
    anomaly_score = 0.1
    
    if voltage > 240 or voltage < 210:
        prediction = "Warning"
        anomaly_score = 0.5
    if temperature > 75:
        prediction = "Critical"
        anomaly_score = 0.9

    return {
        "id": "SUB-001",
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "status": "Operational" if prediction == "Normal" else "Warning",
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "next_maintenance": "2025-12-01"
    }
