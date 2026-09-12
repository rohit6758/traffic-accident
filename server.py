from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import subprocess

app = Flask(__name__)
# Enable CORS so your frontend can communicate with this API
CORS(app)

def load_models():
    # Auto-train models if they don't exist yet
    if not os.path.exists('models/volume_model.pkl') or not os.path.exists('models/risk_model.pkl'):
        print("Training ML models for the first time... please wait a few seconds.")
        subprocess.run(['python', 'train_models.py'], check=True)
        
    vol_mod = joblib.load('models/volume_model.pkl')
    risk_mod = joblib.load('models/risk_model.pkl')
    return vol_mod, risk_mod

volume_model, risk_model = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()
        
        # Extract features
        hour = data.get('hour')
        temperature = data.get('temperature')
        weather_code = data.get('weather_code')
        is_weekend = data.get('is_weekend')
        road_quality = data.get('road_quality')
        
        # Create a DataFrame for the models
        input_data = pd.DataFrame([[hour, temperature, weather_code, is_weekend, road_quality]], 
                                  columns=['hour', 'temperature', 'weather_code', 'is_weekend', 'road_quality'])
        
        # Make predictions
        predicted_vol = volume_model.predict(input_data)[0]
        predicted_risk = risk_model.predict(input_data)[0]
        
        # Send predictions back to frontend as JSON
        return jsonify({
            'success': True,
            'traffic_volume': int(predicted_vol),
            'accident_risk': int(predicted_risk)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("Starting Flask API Server on http://localhost:5000")
    app.run(debug=True, port=5000)
