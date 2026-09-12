import streamlit as st
import pandas as pd
import joblib

import os
import subprocess

# Load models safely
@st.cache_resource
def load_models():
    try:
        # If models don't exist, train them on the fly (useful for cloud deployment)
        if not os.path.exists('models/volume_model.pkl') or not os.path.exists('models/risk_model.pkl'):
            st.warning("Training ML models for the first time... please wait a few seconds.")
            subprocess.run(['python', 'train_models.py'], check=True)
            
        volume_model = joblib.load('models/volume_model.pkl')
        risk_model = joblib.load('models/risk_model.pkl')
        return volume_model, risk_model
    except Exception as e:
        return None, None

volume_model, risk_model = load_models()

# Load summary stats
@st.cache_data
def load_data_stats():
    try:
        df = pd.read_csv('data/traffic_data.csv')
        return len(df), df['traffic_volume'].mean(), df['accident_risk'].mean() * 100
    except:
        return 0, 0, 0

st.set_page_config(page_title="Traffic Intel", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Traffic Analysis"])

if page == "Dashboard":
    st.title("Traffic Volume & Accident Risk Dashboard")
    st.write("Welcome to the Traffic Intelligence System. This ML-powered application analyzes traffic patterns and estimates accident risk.")
    
    total_vehicles, avg_vol, avg_risk = load_data_stats()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Historical Records Analyzed", f"{total_vehicles:,}")
    col2.metric("Average Traffic Volume", f"{int(avg_vol):,}")
    col3.metric("Historical High Risk %", f"{avg_risk:.1f}%")
    
    st.info("Navigate to the **Traffic Analysis** page to input current conditions and get ML predictions.")

elif page == "Traffic Analysis":
    st.title("Analyze Current Conditions")
    
    if volume_model is None or risk_model is None:
        st.error("Models are not trained yet! Please run the Jupyter Notebook first.")
        st.stop()
        
    st.write("Enter the current traffic and weather conditions below:")
    
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Hour of Day", 0, 23, 12)
        temperature = st.number_input("Temperature (°C)", -20, 50, 22)
        weather_code = st.selectbox("Weather Condition", options=[1, 2, 3, 4], format_func=lambda x: {1: "Clear", 2: "Cloudy", 3: "Rain", 4: "Snow"}[x])
    with col2:
        is_weekend = st.radio("Is it the Weekend?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        road_quality = st.selectbox("Road Quality", options=[1, 2, 3], format_func=lambda x: {1: "Good", 2: "Fair", 3: "Poor"}[x])
        
    if st.button("Analyze Traffic & Risk"):
        input_data = pd.DataFrame([[hour, temperature, weather_code, is_weekend, road_quality]], 
                                  columns=['hour', 'temperature', 'weather_code', 'is_weekend', 'road_quality'])
        
        # Predictions
        predicted_vol = volume_model.predict(input_data)[0]
        predicted_risk = risk_model.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        rcol1, rcol2 = st.columns(2)
        rcol1.metric("Predicted Traffic Volume", f"{int(predicted_vol):,} vehicles")
        
        risk_label = "HIGH RISK" if predicted_risk == 1 else "LOW RISK"
        risk_color = "red" if predicted_risk == 1 else "green"
        rcol2.markdown(f"### Accident Risk: <span style='color:{risk_color}'>{risk_label}</span>", unsafe_allow_html=True)
        
        st.warning("Note: This result is an ML-based analytical prediction, not a definitive safety guarantee.")
