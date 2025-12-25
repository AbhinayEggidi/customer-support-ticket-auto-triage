from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_category

# Create API app
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
    category = predict_category(ticket.subject, ticket.description)
    return {"category": category}
