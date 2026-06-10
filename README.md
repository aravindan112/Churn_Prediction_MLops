# Churn_ML_Pipeline-
An end-to-end Machine Learning pipeline for customer churn prediction, built with production-grade MLOps tooling.

![CI](https://github.com/aravindan112/Churn_ML_Pipeline-/actions/workflows/test.yml/badge.svg)

## Overview

This project predicts whether a telecom customer will churn using a full MLOps pipeline — from raw data to a containerized, monitored, and CI/CD-enabled prediction API.

## Architecture
Raw Data → EDA → Preprocessing Pipeline → Model Training (MLflow) → FastAPI Endpoint → Docker Container
↓
Evidently AI Drift Monitoring
↓
GitHub Actions CI/CD

## Tech Stack

| Component | Tool |
|---|---|
| Data Processing | Pandas, NumPy, Scikit-learn |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI + Uvicorn |
| Containerization | Docker |
| Drift Monitoring | Evidently AI |
| CI/CD | GitHub Actions |

## Results

| Model | F1 Score | ROC-AUC | Recall |
|---|---|---|---|
| Logistic Regression | 0.5957 | **0.8407** | 0.5535 |
| Random Forest | 0.5277 | 0.8155 | 0.4706 |
| XGBoost | 0.5894 | 0.8195 | 0.5642 |

**Best model: Logistic Regression (ROC-AUC: 0.8407)**

## Key EDA Findings

- Month-to-month contract customers churn at 42% vs 3% for two-year contracts
- High monthly charges correlate strongly with churn
- Low tenure (0-10 months) customers are the highest churn risk
- Gender has near-zero predictive power

## Project Structure

```
Churn_ML_Pipeline-/
├── data/processed/         # Cleaned and split datasets
├── models/                 # Saved preprocessor and model
├── notebooks/              # EDA, preprocessing, training, drift
├── src/predict.py          # FastAPI prediction endpoint
├── reports/                # Evidently AI drift report
├── Dockerfile
├── requirements.txt
└── .github/workflows/      # CI/CD pipeline
```

## How to Run

### Local
```bash
git clone https://github.com/aravindan112/Churn_ML_Pipeline-.git
cd Churn_ML_Pipeline-
pip install -r requirements.txt
uvicorn src.predict:app --reload
```

### Docker
```bash
docker build -t churn-mlops-api .
docker run -p 8000:8000 churn-mlops-api
```

### API Usage
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

### Response
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.6266,
  "risk_level": "Medium"
}
```

## Author
Aravindan — [GitHub](https://github.com/aravindan112)