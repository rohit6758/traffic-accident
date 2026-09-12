import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import os

# 1. Load data
df = pd.read_csv('data/traffic_data.csv')

# 2. Select Features (X)
X = df[['hour', 'temperature', 'weather_code', 'is_weekend', 'road_quality']]

# 3. Train Volume Model (Regression)
y_volume = df['traffic_volume']
X_train_vol, X_test_vol, y_train_vol, y_test_vol = train_test_split(X, y_volume, test_size=0.2, random_state=42)
volume_model = LinearRegression()
volume_model.fit(X_train_vol, y_train_vol)

# 4. Train Risk Model (Classification)
y_risk = df['accident_risk']
X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X, y_risk, test_size=0.2, random_state=42)
risk_model = LogisticRegression()
risk_model.fit(X_train_risk, y_train_risk)

# 5. Save models
joblib.dump(volume_model, 'models/volume_model.pkl')
joblib.dump(risk_model, 'models/risk_model.pkl')
print("Models trained and saved successfully in traffic-intelligence/models/")
