Vibration-Based Bearing Failure Predictor
FastAPI • Docker • AWS ECS (Fargate) • XGBoost 

This project demonstrates a complete ML deployment workflow for predictive maintenance.
It processes raw high-frequency vibration signals, extracts time/frequency-domain features, scales them, and serves an XGBoost fault classifier through a FastAPI endpoint packaged in Docker and deployed on AWS ECS.

The goal is not to build the “best model”, but to showcase a realistic, production-style ML inference pipeline.

1. Problem Statement

Rotating machinery develops characteristic vibration signatures as bearings degrade.
This project predicts fault vs. non-fault states based on:

- raw time-series signals (.txt files)
- 8 vibration channels
- sampling rates up to 20 MHz
- DSP features (time + frequency domain)

The workflow:
Raw signal → Feature extraction → Scaling → XGBoost inference → REST API response.

2. Repository Structure
vibration-predictor/
│
├── api/
│   ├── app.py                 # FastAPI app with file upload endpoint
│   ├── inference.py            # model loading + prediction
│   ├── extract_features.py     # DSP feature engineering
│   ├── Files.py                # helpers to load .txt multi-column signals
│   └── train.py				# training script
│
├── model/
│   ├── model.json              # trained XGBoost model
│   └── scaler.joblib           # feature scaler
│   
│
├── docker/
│   └── Dockerfile              # container for deployment
│
├── data/
│	├── data_example_1              # small data sample for test
│   ├── data_example_2             
│   └── data_example_3
├── requirements.txt
└── README.md

3. Feature Engineering

Each vibration channel is converted into engineered features:

Time-domain:

RMS
Peak-to-Peak
Zero-crossings

Frequency-domain:

FFT → magnitude spectrum → band energies:
0–1 MHz
1–5 MHz
5–10 MHz
10–20 MHz

Plus:

spectral centroid
dominant frequency (max amplitude)

All channels are processed, features concatenated, and scaled via a trained StandardScaler.
This keeps inference fast, deterministic, and suitable for edge devices or low-latency pipelines.

4. Model

Model type: XGBoost binary classifier
Objective: binary:logistic
Tree method: hist for speed
Training done on pre-extracted features with 80/20 file split.
The data set is IMS bearings dataset from NASA: https://data.nasa.gov/dataset/ims-bearings
The trained model + scaler are stored inside /model.

5. Local API Usage (FastAPI)

Start locally:

uvicorn api.main:app --reload --host 0.0.0.0 --port 8080


Test via curl:

curl -X POST "http://localhost:8080/predict-file" \
  -F "file=@data/example_signal_1.txt" \
  -F "sampling_rate=20000000"


The API returns:

{
  "prediction": value between 0 and 1. If it's below 0.5 it corresponds to the normal state, if it's higher or 0.5 it belongs to a faulty state.
}

6. Docker Usage
Build
docker build -t vibration-api -f Dockerfile .

Run
docker run -p 8080:8080 vibration-api

Test

Same as before:

curl -X POST "http://localhost:8080/predict-file" \
  -F "file=@data/example_signal_1.txt" \
  -F "sampling_rate=20000000"

7. AWS Deployment (ECS + Fargate)

This project is deployed using a production-grade AWS architecture:

AWS Services Used
Service	Purpose
ECR	Docker image registry
ECS Fargate	Serverless container runtime
ALB (Application Load Balancer)	Public endpoint, health checks, routing
VPC + Subnets	Networking & isolation
Security Groups	Traffic control
CloudWatch Logs	Container logs
IAM	Permissions for ECS tasks & ECR pulls

Deployment Flow:

Build image locally
Tag & push to ECR
Create ECS Task Definition
container port: 8080

Create Target Group:

protocol HTTP
port 8080
health check path /health

Create ALB:

port 80 → target group

Create ECS Service:

attach ALB
Fargate launch type
Open port 80 in ALB’s SG
Allow ALB → ECS on port 8080

Once ECS runs the task and ALB marks it healthy, the DNS URL becomes your public inference endpoint.


9. Evaluation & Future Improvements

This project demonstrates:

loading raw sensor data
robust DSP feature extraction
ML model inference
FastAPI-based serving API
production containerization
serverless AWS deployment (ECS Fargate)
load-balanced public endpoint

