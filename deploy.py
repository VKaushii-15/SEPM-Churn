"""
FastAPI Deployment Script - Serves the Churn Prediction Model
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import logging
import pandas as pd
import joblib
import glob
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="E-Commerce Churn API", description="Predict customer churn probability")

# Add CORS so the local index.html can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and reference data
MODEL = None
REFERENCE_DATA = None

def send_slack_notification(customer_id: str, risk_score: float, confidence: float):
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL is not set. Skipping Slack notification.")
        return
        
    try:
        message = {
            "text": f"🚨 *High Risk Churn Alert* 🚨\n*Customer ID*: {customer_id}\n*Risk Score*: {risk_score:.2%}\n*Confidence*: {confidence:.2%}"
        }
        response = requests.post(webhook_url, json=message, timeout=5)
        response.raise_for_status()
        logger.info(f"Slack notification sent for {customer_id}")
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")

class PredictionRequest(BaseModel):
    customerId: str
    accountAge: float
    recency: float
    frequency: float
    monetary: float

@app.on_event("startup")
def load_assets():
    global MODEL, REFERENCE_DATA
    
    # Load Model (get the newest pkl in artifacts)
    model_files = glob.glob("artifacts/churn_model_local_*.pkl")
    if not model_files:
        logger.error("No model found in artifacts/ directory!")
    else:
        latest_model_path = max(model_files, key=os.path.getctime)
        logger.info(f"Loading {latest_model_path}...")
        MODEL = joblib.load(latest_model_path)
    
    # Load default data to fill missing 16 features
    try:
        df = pd.read_csv("sample_customers.csv")
        logger.info("Loaded reference dataset for imputation.")
    except Exception as e:
        df = pd.DataFrame()
        logger.warning(f"Could not load sample_customers.csv: {e}")
        
    REFERENCE_DATA = df

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        raise HTTPException(status_code=404, detail="Frontend file not found.")

@app.post("/predict")
def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    # Search for user in Reference Data
    user_data = None
    if REFERENCE_DATA is not None and not REFERENCE_DATA.empty and 'customer_id' in REFERENCE_DATA.columns:
        matching_rows = REFERENCE_DATA[REFERENCE_DATA['customer_id'] == request.customerId]
        if not matching_rows.empty:
            user_data = matching_rows.iloc[0].to_dict()
            
    # If not found, use a median/mean or zero profile from the reference dataset
    if user_data is None:
        if REFERENCE_DATA is not None and not REFERENCE_DATA.empty:
            user_data = REFERENCE_DATA.drop(['customer_id', 'churn'], axis=1, errors='ignore').median().to_dict()
        else:
            raise HTTPException(status_code=500, detail="No reference data available to impute missing features.")

    # Override the user_data with the 4 fields from frontend
    user_data['account_age_days'] = request.accountAge
    user_data['days_since_last_purchase'] = request.recency
    if 'purchase_count_30d' in user_data: user_data['purchase_count_30d'] = request.frequency
    if 'total_spend_30d' in user_data: user_data['total_spend_30d'] = request.monetary
    
    # Prepare dataframe for prediction
    # Ensure columns match training order exactly: drop id and churn
    if 'customer_id' in user_data: del user_data['customer_id']
    if 'churn' in user_data: del user_data['churn']
    
    # Form a single-row DataFrame
    input_df = pd.DataFrame([user_data])
    
    try:
        # Get probability of class 1
        churn_prob = float(MODEL.predict_proba(input_df)[0, 1])
        prediction = 1 if churn_prob > 0.5 else 0
        confidence = abs(churn_prob - 0.5) * 2
        
        if prediction == 1:
            send_slack_notification(request.customerId, churn_prob, confidence)
            
        return {
            "churnProbability": churn_prob,
            "churnPrediction": prediction,
            "confidence": confidence,
            "message": "Success"
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("deploy:app", host="0.0.0.0", port=port)
