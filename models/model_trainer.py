"""
ML model training pipeline with MLflow tracking
"""

import logging
from typing import Tuple, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MLFLOW_TRACKING_URI = "file:./mlruns"
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_N_ESTIMATORS = 300
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
OPTUNA_N_TRIALS = 50


class ModelTrainer:
    """ML model training and evaluation"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("churn_prediction_experiment")

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = RANDOM_STATE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        logger.info("Features scaled using StandardScaler")
        return X_train_scaled, X_val_scaled, X_test_scaled

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        params: Dict = None
    ) -> xgb.XGBClassifier:
        if params is None:
            params = {
                'max_depth': XGB_MAX_DEPTH,
                'learning_rate': XGB_LEARNING_RATE,
                'n_estimators': XGB_N_ESTIMATORS,
                'subsample': XGB_SUBSAMPLE,
                'colsample_bytree': XGB_COLSAMPLE_BYTREE,
                'random_state': RANDOM_STATE,
                'eval_metric': 'logloss',
                'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),
                'tree_method': 'hist'
            }
        params['early_stopping_rounds'] = 50
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        logger.info("XGBoost model trained")
        return model

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        params: Dict = None
    ) -> lgb.LGBMClassifier:
        if params is None:
            params = {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_STATE,
                'metric': 'binary_logloss',
                'is_unbalanced': True,
                'verbose': -1
            }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=-1)]
        )
        logger.info("LightGBM model trained")
        return model

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        params: Dict = None
    ) -> RandomForestClassifier:
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("Random Forest model trained")
        return model

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, float]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        logger.info(f"{model_name} - F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")
        return metrics

    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        model_type: str = 'xgboost',
        n_trials: int = OPTUNA_N_TRIALS
    ) -> Dict[str, Any]:
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                    'random_state': RANDOM_STATE,
                    'eval_metric': 'logloss',
                    'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),
                    'tree_method': 'hist'
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif model_type == 'lightgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                    'random_state': RANDOM_STATE,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(period=-1)])
            y_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_proba)

        sampler = TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=MedianPruner())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")
        return best_params

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        with mlflow.start_run(run_name=f"ensemble_{datetime.now().isoformat()}"):
            ensemble_results = {}
            logger.info("Training XGBoost model...")
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
            ensemble_results['xgboost'] = {'model': xgb_model, 'metrics': self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")}
            logger.info("Training LightGBM model...")
            lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
            ensemble_results['lightgbm'] = {'model': lgb_model, 'metrics': self.evaluate_model(lgb_model, X_test, y_test, "LightGBM")}
            logger.info("Training Random Forest model...")
            rf_model = self.train_random_forest(X_train, y_train)
            ensemble_results['random_forest'] = {'model': rf_model, 'metrics': self.evaluate_model(rf_model, X_test, y_test, "Random Forest")}
            best_model_name = max(ensemble_results.keys(), key=lambda k: ensemble_results[k]['metrics']['pr_auc'])
            self.best_model = ensemble_results[best_model_name]['model']
            best_metrics = ensemble_results[best_model_name]['metrics']
            logger.info(f"Best model: {best_model_name} with PR-AUC: {best_metrics['pr_auc']:.4f}")
            for model_name, result in ensemble_results.items():
                for metric_name, metric_value in result['metrics'].items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_pr_auc", best_metrics['pr_auc'])
            if isinstance(self.best_model, xgb.XGBClassifier):
                mlflow.xgboost.log_model(self.best_model, "model")
            else:
                mlflow.sklearn.log_model(self.best_model, "model")
            return {
                'best_model': self.best_model,
                'best_model_name': best_model_name,
                'metrics': best_metrics,
                'all_results': ensemble_results
            }

    def extract_feature_importance(self, model, feature_names=None) -> pd.DataFrame:
        if isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier, RandomForestClassifier)):
            importance = model.feature_importances_
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance))]
        else:
            return pd.DataFrame()
        return pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)

    def save_model(self, model, path: str):
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        model = joblib.load(path)
        self.best_model = model
        logger.info(f"Model loaded from {path}")
        return model


class ModelRegistry:
    """MLflow model registry management"""

    def __init__(self):
        self.client = mlflow.tracking.MlflowClient()

    def register_model(self, run_id: str, model_name: str, stage: str = "Staging") -> str:
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered: {model_name}")
        return result.model_version

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
        logger.info(f"Model {model_name}:{version} transitioned to {stage}")

    def get_production_model(self, model_name: str):
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0]
        return None


if __name__ == "__main__":
    trainer = ModelTrainer()
    # X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    # X_train_s, X_val_s, X_test_s = trainer.scale_features(X_train, X_val, X_test)
    # results = trainer.train_ensemble(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test)