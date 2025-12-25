# Load model and predict category

import pickle

# Load trained model
with open("model/ticket_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_category(subject, description):
    text = subject + " " + description
    return model.predict([text])[0]
