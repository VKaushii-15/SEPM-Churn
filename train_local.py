"""
Local ML Model Training Script (Offline)
Trains the full classification ensemble with sample_customers.csv.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import logging
import pandas as pd
import warnings
from datetime import datetime

# Import local modules
from models.model_trainer import ModelTrainer
import mlflow

# Suppress mlflow connection warnings if any
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*60)
    logger.info("STARTING LOCAL OFFLINE MODEL TRAINING")
    logger.info("="*60)
    
    # 1. Override tracking URI to ensure it works locally
    mlflow.set_tracking_uri(f"file:///{os.path.abspath('./mlruns')}")
    
    # 2. Get data
    logger.info("Loading sample_customers.csv...")
    data = pd.read_csv('sample_customers.csv')
    
    # Drop customer_id and isolate target 'churn'
    if 'churn' not in data.columns:
        raise ValueError("Target column 'churn' not found in dataset.")
        
    labels = data['churn']
    features = data.drop(['customer_id', 'churn'], axis=1, errors='ignore')
    
    logger.info(f"Class distribution: {labels.value_counts(normalize=True).to_dict()}")
    
    # 3. Train models
    trainer = ModelTrainer()
    
    logger.info("Splitting data...")
    try:
        if len(features) < 100:
             logger.warning("Dataset is very small! Adjusting splits to prevent errors.")
             X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(
                 features,
                 labels,
                 test_size=0.1,
                 val_size=0.1
             )
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(
                features,
                labels
            )
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        raise
        
    logger.info("Skipping scaling for raw tree-based model features...")
    
    logger.info("Training ensemble models (XGBoost, LightGBM, Random Forest)...")
    results = trainer.train_ensemble(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )
    
    logger.info("="*60)
    logger.info(f"BEST MODEL FOUND: {results['best_model_name']}")
    for metric, val in results['metrics'].items():
        logger.info(f"  {metric}: {val:.4f}")
    
    # 4. Save best model artifact locally
    os.makedirs("artifacts", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"artifacts/churn_model_local_{timestamp}.pkl"
    trainer.save_model(results['best_model'], model_path)
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE! 🚀")
    logger.info(f"Model saved to: {os.path.abspath(model_path)}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
