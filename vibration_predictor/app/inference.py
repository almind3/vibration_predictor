import xgboost as xgb
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model", "xgb_model.json")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.joblib")

def load_model():
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    return booster

def load_scaler():
    scaler = joblib.load(SCALER_PATH)
    return scaler

def predict_proba(model, X_scaled):
    dtest = xgb.DMatrix(X_scaled)
    preds = model.predict(dtest)
    return preds