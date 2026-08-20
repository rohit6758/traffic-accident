<h1 align="center">
  🚗 Traffic Intelligence System (AI)
</h1>

<p align="center">
  <strong>Advanced Machine Learning Portal for Smart City Traffic Forecasting & Accident Risk Triage</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Backend-Flask-green.svg?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Frontend-Vanilla_JS-yellow.svg?style=for-the-badge&logo=javascript" alt="JS">
</p>

---

## 🚦 Project Overview

The **Traffic Intelligence System** is an end-to-end full-stack Artificial Intelligence application designed for modern urban planning and emergency dispatch routing. By taking environmental parameters (Time, Temperature, Weather, and Road Quality), the system executes rapid inference using pre-trained regression algorithms.

### 🌟 Core Features

- **📊 Continuous Volume Prediction:** Employs **Linear Regression** to forecast the exact influx of vehicles on a given road based on temporal and environmental heuristics.
- **⚠️ Binary Accident Triage:** Utilizes **Logistic Regression** to compute real-time accident risk (Low/High), acting as an early warning system for emergency responders.
- **⚡ Ultra-Low Latency Inference:** The ML models are serialized using `joblib` and globally loaded into the Flask server's RAM upon startup, allowing the API to respond to frontend requests in milliseconds.
- **📱 Dynamic Stateless Dashboard:** A clean, visually accessible HTML/JS frontend styled with CSS variables and powered entirely by asynchronous Fetch API calls—requiring absolutely zero page reloads.

---

## 🛠️ System Architecture

```mermaid
graph TD;
    A[Frontend Dashboard<br>(HTML/CSS/JS)] -->|JSON Payload<br>Fetch API POST| B(Flask REST API<br>server.py)
    B --> C{joblib Load}
    C -->|Linear Regression| D[volume_model.pkl]
    C -->|Logistic Regression| E[risk_model.pkl]
    D --> B
    E --> B
    B -->|Predicted Volume & Risk| A
```

---

## 📂 Repository Structure

```text
traffic-accident/
│
├── index.html                      # Real-time traffic dashboard UI
├── requirements.txt                # Deployment dependencies
├── server.py                       # High-performance Flask REST API
├── train_models.py                 # Scikit-Learn ML generation & training script
│
├── models/                         # Serialized AI model artifacts
│   ├── risk_model.pkl              
│   └── volume_model.pkl            
│
└── Traffic_Project_Report.pdf      # Detailed Official Project Documentation
```

---

## 🚀 Quick Start Execution Guide

### 1. Environment Setup
Clone the repository and install the strict dependencies required for the ML engine:
```bash
git clone https://github.com/rohit6758/traffic-accident.git
cd traffic-accident
pip install -r requirements.txt
```

### 2. Generate and Train Models
Execute the core AI script. This synthesizes a massive 50,000-row environmental dataset, fits the regressions, and serializes the state to the `/models` directory:
```bash
python train_models.py
```

### 3. Ignite the Backend API
Start the Flask server to open the JSON inference routes:
```bash
python server.py
```
> *Server will successfully bind to `http://127.0.0.1:5000`*

### 4. Access the Dashboard
Double-click `index.html` in your file explorer, or serve it on a local port. Select the environmental variables and instantly retrieve the AI's diagnostic results!

---

<p align="center">
  <i>Developed with precision by <b>A. Rohit</b> for the APSSDC-Summer Online Internship.</i>
</p>
