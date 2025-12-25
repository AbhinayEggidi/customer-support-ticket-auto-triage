from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load trained model
with open("model/ticket_model.pkl", "rb") as f:
    model = pickle.load(f)

# FastAPI app
app = FastAPI(title="Customer Support Ticket Auto-Triage")

# Request format
class Ticket(BaseModel):
    subject: str
    description: str

# Health check
@app.get("/")
def home():
    return {"message": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(ticket: Ticket):
    text = ticket.subject + " " + ticket.description
    prediction = model.predict([text])[0]
    return {"category": prediction}
