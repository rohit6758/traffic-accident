import pandas as pd
import numpy as np

# Generate Massive Realistic Traffic Data (50,000 rows)
np.random.seed(42)
num_samples = 50000

# Features
hour = np.random.randint(0, 24, num_samples)
temperature = np.random.normal(20, 10, num_samples) # Celsius
weather_code = np.random.choice([1, 2, 3, 4], num_samples, p=[0.5, 0.3, 0.15, 0.05]) # 1: Clear, 2: Cloudy, 3: Rain, 4: Snow
is_weekend = np.where((hour % 168) >= 120, 1, 0)
road_quality = np.random.choice([1, 2, 3], num_samples, p=[0.6, 0.25, 0.15]) # 1: Good, 2: Fair, 3: Poor

# Target 1: Traffic Volume
volume_base = 1200
rush_hour_multiplier = np.where(((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18)), 2.8, 1.0)
night_multiplier = np.where((hour >= 23) | (hour <= 5), 0.2, 1.0)
weekend_multiplier = np.where(is_weekend == 1, 0.7, 1.0) # Less traffic on weekends generally
weather_volume_penalty = np.where(weather_code >= 3, 0.75, 1.0) # Less people drive in bad weather

traffic_volume = (volume_base * rush_hour_multiplier * night_multiplier * weekend_multiplier * weather_volume_penalty) + np.random.normal(0, 150, num_samples)
traffic_volume = np.clip(traffic_volume, 50, 5000).astype(int)

# Target 2: Accident Risk (Classification: 0=Low, 1=High)
# Heavily penalize poor roads and bad weather
risk_prob = np.full(num_samples, 0.02) # Base risk 2%

# Add risk based on weather
risk_prob = np.where(weather_code == 3, risk_prob + 0.2, risk_prob) # Rain adds 20%
risk_prob = np.where(weather_code == 4, risk_prob + 0.5, risk_prob) # Snow adds 50%

# Add risk based on roads
risk_prob = np.where(road_quality == 2, risk_prob + 0.15, risk_prob) # Fair roads add 15%
risk_prob = np.where(road_quality == 3, risk_prob + 0.60, risk_prob) # Poor roads add 60% risk!

# High traffic volume adds risk
risk_prob = np.where(traffic_volume > 3500, risk_prob + 0.15, risk_prob)

accident_risk = np.random.binomial(1, np.clip(risk_prob, 0, 1))

df = pd.DataFrame({
    'hour': hour,
    'temperature': np.round(temperature, 1),
    'weather_code': weather_code,
    'is_weekend': is_weekend,
    'road_quality': road_quality,
    'traffic_volume': traffic_volume,
    'accident_risk': accident_risk
})

df.to_csv('data/traffic_data.csv', index=False)
print("Massive dataset (50k rows) generated successfully!")
