Vibration-Based Bearing Failure Predictor <br />
FastAPI • Docker • AWS ECS (Fargate) • XGBoost  <br />
<br />
This project demonstrates a complete ML deployment workflow for predictive maintenance.<br />
It processes raw high-frequency vibration signals, extracts time/frequency-domain features, scales them, and serves an XGBoost fault classifier through a FastAPI endpoint packaged in Docker and deployed on AWS ECS.<br />
<br />
The goal is not to build the “best model”, but to showcase a realistic, production-style ML inference pipeline.<br />
<br />
1. Problem Statement<br />
<br />
Rotating machinery develops characteristic vibration signatures as bearings degrade.<br />
This project predicts fault vs. non-fault states based on:<br />
<br />
- raw time-series signals (.txt files)<br />
- 8 vibration channels<br />
- sampling rates up to 20 MHz<br />
- DSP features (time + frequency domain)<br />
<br />
The workflow:<br />
Raw signal → Feature extraction → Scaling → XGBoost inference → REST API response.<br />
<br />
2. Repository Structure <br />
vibration-predictor/ <br />
│ <br />
├── api/ <br />
│   ├── app.py                    # FastAPI app with file upload endpoint     <br />
│   ├── inference.py               # model loading + prediction <br />
│   ├── extract_features.py        # DSP feature engineering <br />
│   ├── Files.py                   # helpers to load .txt multi-column signals <br />
│   └── train.py			           	# training script<br />
│<br />
├── model/ <br />
│   ├── model.json                 # trained XGBoost model<br />
│   └── scaler.joblib              # feature scaler<br />
│   <br />
│<br />
├── docker/<br />
│   └── Dockerfile                 # container for deployment<br />
│<br />
├── data/<br />
│	├── data_example_1                 # small data sample for test<br />
│   ├── data_example_2             <br />
│   └── data_example_3<br />
├── requirements.txt<br />
└── README.md<br />
<br />
3. Feature Engineering<br />
<br />
Each vibration channel is converted into engineered features:<br />
<br />
Time-domain:<br />
<br />
RMS<br />
Peak-to-Peak<br />
Zero-crossings<br />
<br />
Frequency-domain:<br />
<br />
FFT → magnitude spectrum → band energies:<br />
0–1 MHz<br />
1–5 MHz<br />
5–10 MHz<br />
10–20 MHz<br />
<br />
Plus:<br />
<br />
spectral centroid<br />
dominant frequency (max amplitude)<br />
<br />
All channels are processed, features concatenated, and scaled via a trained StandardScaler.<br />
This keeps inference fast, deterministic, and suitable for edge devices or low-latency pipelines.<br />
<br />
4. Model<br />
<br />
Model type: XGBoost binary classifier<br />
Objective: binary:logistic<br />
Tree method: hist for speed<br />
Training done on pre-extracted features with 80/20 file split.<br />
The data set is IMS bearings dataset from NASA: https://data.nasa.gov/dataset/ims-bearings<br />
The trained model + scaler are stored inside /model.<br />
<br />
5. Local API Usage (FastAPI)<br />
<br />
Start locally:<br />
<br />
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080<br />
<br />
<br />
Test via curl:<br />
<br />
curl -X POST "http://localhost:8080/predict-file" \<br />
  -F "file=@data/example_signal_1.txt" \<br />
  -F "sampling_rate=20000000"<br />
<br />
<br />
The API returns:<br />
<br />
{<br />
  "prediction": value between 0 and 1. If it's below 0.5 it corresponds to the normal state, if it's higher or 0.5 it belongs to a faulty state.<br />
}<br />
<br />
6. Docker Usage<br />
Build<br />
docker build -t vibration-api -f Dockerfile .<br />
<br />
Run<br />
docker run -p 8080:8080 vibration-api<br />
<br />
Test<br />
<br />
Same as before:<br />
<br />
curl -X POST "http://localhost:8080/predict-file" \
  -F "file=@data/example_signal_1.txt" \
  -F "sampling_rate=20000000"
<br />
7. AWS Deployment (ECS + Fargate)<br />
<br />
This project is deployed using a production-grade AWS architecture:<br />
<br />
AWS Services Used<br />
Service	Purpose<br />
ECR	Docker image registry<br />
ECS Fargate	Serverless container runtime<br />
ALB (Application Load Balancer)	Public endpoint, health checks, routing<br />
VPC + Subnets	Networking & isolation<br />
Security Groups	Traffic control<br />
CloudWatch Logs	Container logs<br />
IAM	Permissions for ECS tasks & ECR pulls<br />
<br />
Deployment Flow:<br />
<br />
Build image locally<br />
Tag & push to ECR<br />
Create ECS Task Definition<br />
container port: 8080<br />
<br />
Create Target Group:<br />
<br />
protocol HTTP<br />
port 8080<br />
health check path /health<br />
<br />
Create ALB:<br />
<br />
port 80 → target group<br />
<br />
Create ECS Service:<br />
<br />
attach ALB<br />
Fargate launch type<br />
Open port 80 in ALB’s SG<br />
Allow ALB → ECS on port 8080<br />
<br />
Once ECS runs the task and ALB marks it healthy, the DNS URL becomes your public inference endpoint.<br />
<br />

9. Evaluation & Future Improvements<br />
<br />
This project demonstrates:<br />
<br />
loading raw sensor data<br />
robust DSP feature extraction<br />
ML model inference<br />
FastAPI-based serving API<br />
production containerization<br />
serverless AWS deployment (ECS Fargate)
load-balanced public endpoint


