# customer-support-ticket-auto-triage
Machine learning based customer support ticket classification and auto-triage system using FastAPI.

# Customer Support Ticket Auto-Triage

## Project Overview
This project automates the classification of customer support tickets using machine learning.
It reduces manual effort and improves ticket routing efficiency.

## Ticket Categories
- Bug Report
- Feature Request
- Technical Issue
- Billing Inquiry
- Account Management

## Tech Stack
- Python 3.11
- scikit-learn
- pandas
- FastAPI

## How to Run

## Install Dependencies
 
pip install -r requirements.txt

**Train the Model**

First, open the terminal and run the training script:

python src/train_model.py

**Activate the API**

python -m uvicorn src.api:app --reload


Once the API is running, add endpoint docs endpoint to api open the Swagger UI, select the /predict endpoint, click Try it out, and enter the subject and description of the ticket to get the predicted category.






 
