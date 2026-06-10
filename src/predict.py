from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI(title="Churn Prediction API")

# Load preprocessor and model
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    import pandas as pd
    input_df = pd.DataFrame([data.dict()])
    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }