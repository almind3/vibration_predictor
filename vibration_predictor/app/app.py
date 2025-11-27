from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import pandas as pd
from app.extract_features import extract_features
from app.inference import load_model, predict_proba, load_scaler

app = FastAPI()
model = load_model()
scaler = load_scaler()


@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    sampling_rate: float = Form(...)
):
    # Read raw file contents
    content = await file.read()

    df = pd.read_csv(
        filepath_or_buffer=pd.io.common.StringIO(content.decode()),
        sep="\t",
        header=None,
        dtype=np.float32
    )
    data = df.values
    channel_N = data.shape[1]
    feat = []
    for jj in range(channel_N):
        tmp = extract_features.extract_features(data[:, jj], sampling_rate)
        feat.append(tmp)

    features_final = np.concatenate(feat, axis=0)
    features_final = features_final.reshape(1, -1)
    features_final = scaler.transform(features_final)
    # Predict
    proba = predict_proba(model, features_final)

    return {"fault_probability": float(proba[0])}