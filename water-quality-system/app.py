# main application
from generate_water_dataset import generate_water_data
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import datetime
import io
import re
from collections import defaultdict
import sqlite3
import os
import requests
import json
from math import radians, sin, cos, sqrt, atan2
import folium
from streamlit_folium import folium_static
import openpyxl
import time
from bluetooth_utils import BluetoothManager, SensorDataCollector
import tempfile

# Set page configuration with wider layout
st.set_page_config(
    page_title="Water Quality Prediction System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set larger font sizes for matplotlib
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect("water_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        location TEXT,
        pH REAL,
        turbidity REAL,
        TDS REAL,
        residual_chlorine REAL,
        temperature REAL,
        rainfall REAL,
        water_source_type TEXT,
        water_treatment TEXT,
        sanitation_facilities INTEGER,
        open_defecation INTEGER,
        waste_management INTEGER,
        risk_score REAL,
        water_safety TEXT,
        cholera_risk REAL,
        typhoid_risk REAL,
        dysentery_risk REAL,
        UNIQUE(timestamp, location)
    )
    """)
    conn.commit()
    conn.close()

init_db()

# Weather API function with Excel data integration
def get_weather_data(location, date=None):
    try:
        # Check if we have Excel weather data uploaded
        if 'weather_excel_data' in st.session_state and not st.session_state.weather_excel_data.empty:
            excel_data = st.session_state.weather_excel_data
            if location in excel_data['location'].values:
                location_data = excel_data[excel_data['location'] == location]
                if date and 'date' in location_data.columns:
                    date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime.date) else str(date)
                    location_data = location_data[location_data['date'] == date_str]
                
                if not location_data.empty:
                    # Get the latest record for the location
                    record = location_data.iloc[-1]
                    return {
                        'rainfall': record.get('rainfall', np.random.uniform(0, 25)),
                        'temperature': record.get('temperature', np.random.uniform(15, 30)),
                        'humidity': record.get('humidity', np.random.uniform(60, 95))
                    }
        
        # Fallback to API or mock data
        API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "demo_key")
        BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
        
        # For demonstration, return mock data
        return {
            'rainfall': np.random.uniform(0, 25),
            'temperature': np.random.uniform(15, 30),
            'humidity': np.random.uniform(60, 95)
        }
    except:
        # Fallback to mock data if API fails
        return {
            'rainfall': np.random.uniform(0, 25),
            'temperature': np.random.uniform(15, 30),
            'humidity': np.random.uniform(60, 95)
        }

# Title and description with larger font
st.title("💧 Comprehensive Water Quality Prediction System")
st.markdown("""
<div style="font-size: 18px;">
This system predicts water-borne diseases in rural areas using multiple data sources:
1. Sensor data analysis (pH, turbidity, TDS, residual chlorine, temperature)
2. Weather data (rainfall)
3. Sanitation survey data
4. Symptom analysis through natural language processing
5. Fusion of all approaches for enhanced accuracy
</div>
""", unsafe_allow_html=True)

# Initialize session state for data storage
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = pd.DataFrame()
if 'symptom_data' not in st.session_state:
    st.session_state.symptom_data = pd.DataFrame()
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = pd.DataFrame()
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = pd.DataFrame()
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = None
if 'village_coordinates' not in st.session_state:
    st.session_state.village_coordinates = {}
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'sensor_status' not in st.session_state:
    st.session_state.sensor_status = {}
if 'bluetooth_manager' not in st.session_state:
    st.session_state.bluetooth_manager = BluetoothManager()
if 'data_collector' not in st.session_state:
    st.session_state.data_collector = SensorDataCollector()
if 'weather_excel_data' not in st.session_state:
    st.session_state.weather_excel_data = pd.DataFrame()

# Load generated dataset
@st.cache_data
def load_generated_data():
    file_path = "Generate_data_set/indian_tribal_water_metrics_2025.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        # Generate the dataset if it doesn't exist
        st.info("Generating dataset...")
        df = generate_water_data()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

# Symptom keyword mapping
symptom_keywords = {
    'cholera': ['vomiting', 'diarrhea', 'dehydration', 'rice water stools', 'leg cramps'],
    'typhoid': ['fever', 'headache', 'abdominal pain', 'constipation', 'diarrhea', 'rose spots'],
    'dysentery': ['bloody diarrhea', 'abdominal cramps', 'fever', 'nausea'],
    'general': ['fever', 'nausea', 'stomach pain', 'cramps', 'dehydration']
}

# Function to process natural language symptoms
def process_symptoms(text):
    symptoms = defaultdict(int)
    text = text.lower()
    
    for disease, keywords in symptom_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                symptoms[disease] += 1
                
    # Calculate risk scores based on symptom frequency
    total_symptoms = sum(symptoms.values())
    if total_symptoms > 0:
        return {
            'cholera_risk': symptoms['cholera'] / total_symptoms * 100,
            'typhoid_risk': symptoms['typhoid'] / total_symptoms * 100,
            'dysentery_risk': symptoms['dysentery'] / total_symptoms * 100
        }
    return {'cholera_risk': 0, 'typhoid_risk': 0, 'dysentery_risk': 0}

# Function to calculate distance between two locations (simplified)
def calculate_distance(loc1, loc2):
    # Simplified distance calculation - in real implementation, use geocoordinates
    return np.random.uniform(1, 20)

# Function to upload and process Excel files
def process_uploaded_file(uploaded_file, data_type):
    try:
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Process based on data type
        if data_type == "weather":
            # Ensure required columns are present
            required_cols = ['location', 'rainfall']
            if all(col in df.columns for col in required_cols):
                if 'date' not in df.columns:
                    df['date'] = datetime.date.today()
                return df
            else:
                st.error(f"Weather data must contain these columns: {required_cols}")
                return None
                
        elif data_type == "sensor":
            required_cols = ['location', 'pH', 'turbidity', 'TDS']
            if all(col in df.columns for col in required_cols):
                if 'timestamp' not in df.columns:
                    df['timestamp'] = datetime.datetime.now()
                return df
            else:
                st.error(f"Sensor data must contain these columns: {required_cols}")
                return None
                
        elif data_type == "symptom":
            required_cols = ['location', 'symptoms_text']
            if all(col in df.columns for col in required_cols):
                if 'timestamp' not in df.columns:
                    df['timestamp'] = datetime.datetime.now()
                return df
            else:
                st.error(f"Symptom data must contain these columns: {required_cols}")
                return None
                
        elif data_type == "survey":
            required_cols = ['location', 'water_source_type', 'sanitation_facilities']
            if all(col in df.columns for col in required_cols):
                if 'timestamp' not in df.columns:
                    df['timestamp'] = datetime.datetime.now()
                return df
            else:
                st.error(f"Survey data must contain these columns: {required_cols}")
                return None
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Sidebar for navigation with larger font
st.sidebar.markdown("<h2 style='font-size: 24px;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Home", "Data Generation & Sensors", "Sensor Data", "Symptom Analysis", "Weather Data", 
                                 "Sanitation Survey", "Village Coordinates", "Fusion Analysis", 
                                 "Predictive Analytics", "Risk Dashboard", "Regional Analysis"])

# Home page
if page == "Home":
    st.header("Welcome to the Water Quality Prediction System")
    st.markdown("""
    <div style="font-size: 18px;">
    This system helps predict water-borne disease risks in rural Northeast India by analyzing multiple data sources.
    
    <b>Key Features:</b>
    - Water quality sensor data analysis
    - Symptom reporting and analysis
    - Weather data integration
    - Sanitation survey data collection
    - Predictive analytics for disease outbreaks
    - Regional risk mapping
    
    <b>How to use:</b>
    1. Start by adding village coordinates in the 'Village Coordinates' section
    2. Input sensor data, weather data, and survey data for each village
    3. Analyze symptoms reported by community health workers
    4. View predictions and risk assessments in the dashboard
    </div>
    """, unsafe_allow_html=True)
    
    # Display sample data
    if st.button("Load Sample Data"):
        sample_data = load_generated_data()
        st.session_state.sensor_data = sample_data
        st.success(f"Loaded {len(sample_data)} records of sample data")
        
        # Show basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Villages Covered", sample_data['location'].nunique())
        with col2:
            st.metric("Time Period", f"{sample_data['timestamp'].min().date()} to {sample_data['timestamp'].max().date()}")
        with col3:
            avg_risk = sample_data['risk_score'].mean()
            st.metric("Average Risk Score", f"{avg_risk:.2f}")

# Data Generation & Sensors page
elif page == "Data Generation & Sensors":
    st.header("Data Generation & Sensor Management")
    
    # Form for data generation parameters
    with st.expander("Data Generation Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            n_villages = st.slider("Number of Villages", 1, 20, 5)
            n_days = st.slider("Number of Days to Generate", 1, 365, 30)
        with col2:
            n_water_sources = st.slider("Number of Water Sources", 1, 10, 3)
            include_weather = st.checkbox("Include Weather Data", True)
        
        if st.button("Generate Data"):
            with st.spinner("Generating data..."):
                # Generate the dataset
                df = generate_water_data(
                    n_villages=n_villages,
                    n_days=n_days,
                    n_water_sources=n_water_sources,
                    include_weather=include_weather
                )
                st.session_state.generated_data = df
                st.success(f"Generated {len(df)} records for {n_villages} villages over {n_days} days")
    
    # Display generated data if available
    if st.session_state.generated_data is not None:
        st.subheader("Generated Data Preview")
        st.dataframe(st.session_state.generated_data.head())
        
        # Option to save generated data to main dataset
        if st.button("Add to Main Dataset"):
            if st.session_state.sensor_data.empty:
                st.session_state.sensor_data = st.session_state.generated_data
            else:
                st.session_state.sensor_data = pd.concat(
                    [st.session_state.sensor_data, st.session_state.generated_data], 
                    ignore_index=True
                )
            st.success("Generated data added to main dataset!")
    
    # Sensor connection and status section
    st.subheader("Sensor Connection & Status")
    
    # Bluetooth connection interface
    with st.expander("Bluetooth Sensor Connection"):
        st.info("This section will connect to physical sensors when hardware is available")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Scan for Sensors"):
                # Simulate scanning for sensors
                with st.spinner("Scanning for sensors..."):
                    time.sleep(2)
                    # Simulate found sensors
                    simulated_sensors = {
                        "pH_Sensor_001": {"battery": 85, "condition": "Good", "connected": False, "type": "pH"},
                        "Turbidity_Sensor_002": {"battery": 72, "condition": "Fair", "connected": False, "type": "turbidity"},
                        "TDS_Sensor_003": {"battery": 90, "condition": "Excellent", "connected": False, "type": "TDS"},
                        "Chlorine_Sensor_004": {"battery": 65, "condition": "Needs Maintenance", "connected": False, "type": "chlorine"},
                        "Temperature_Sensor_005": {"battery": 80, "condition": "Good", "connected": False, "type": "temperature"}
                    }
                    st.session_state.sensor_status = simulated_sensors
                    st.success(f"Found {len(simulated_sensors)} sensors")
        
        with col2:
            if st.button("Connect to All Sensors"):
                if st.session_state.sensor_status:
                    # Simulate connecting to sensors
                    for sensor_id in st.session_state.sensor_status:
                        st.session_state.sensor_status[sensor_id]["connected"] = True
                    st.success("Connected to all available sensors")
                else:
                    st.warning("No sensors found. Please scan for sensors first.")
    
    # Display sensor status
    if st.session_state.sensor_status:
        st.subheader("Sensor Status")
        
        # Create a DataFrame for better display
        sensor_df = pd.DataFrame.from_dict(st.session_state.sensor_status, orient='index')
        st.dataframe(sensor_df)
        
        # Visualize battery status
        fig, ax = plt.subplots(figsize=(10, 6))
        sensors = list(st.session_state.sensor_status.keys())
        battery_levels = [st.session_state.sensor_status[s]["battery"] for s in sensors]
        
        colors = []
        for level in battery_levels:
            if level > 80:
                colors.append('green')
            elif level > 60:
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = ax.bar(sensors, battery_levels, color=colors)
        ax.set_ylabel('Battery Level (%)')
        ax.set_title('Sensor Battery Status')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, battery_levels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{value}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Automatic data collection from sensors
        if st.button("Collect Data from Sensors"):
            connected_sensors = [s for s in st.session_state.sensor_status if st.session_state.sensor_status[s]["connected"]]
            if connected_sensors:
                # Simulate data collection from connected sensors
                with st.spinner("Collecting data from sensors..."):
                    time.sleep(3)
                    
                    # Generate sample data from sensors based on type
                    sensor_readings = {'timestamp': datetime.datetime.now()}
                    
                    for sensor_id in connected_sensors:
                        sensor_type = st.session_state.sensor_status[sensor_id]["type"]
                        
                        if sensor_type == "pH":
                            sensor_readings['pH'] = round(np.random.uniform(6.5, 8.5), 2)
                        elif sensor_type == "turbidity":
                            sensor_readings['turbidity'] = round(np.random.uniform(0.1, 15.0), 2)
                        elif sensor_type == "TDS":
                            sensor_readings['TDS'] = round(np.random.uniform(100, 800), 1)
                        elif sensor_type == "chlorine":
                            sensor_readings['residual_chlorine'] = round(np.random.uniform(0.1, 2.0), 2)
                        elif sensor_type == "temperature":
                            sensor_readings['temperature'] = round(np.random.uniform(15, 35), 1)
                    
                    # Display collected data
                    st.subheader("Collected Sensor Data")
                    st.json(sensor_readings)
                    
                    # Option to add to main dataset
                    if st.button("Add to Main Dataset"):
                        new_data = pd.DataFrame([sensor_readings])
                        if st.session_state.sensor_data.empty:
                            st.session_state.sensor_data = new_data
                        else:
                            st.session_state.sensor_data = pd.concat(
                                [st.session_state.sensor_data, new_data], 
                                ignore_index=True
                            )
                        st.success("Sensor data added to main dataset!")
            else:
                st.warning("No sensors connected. Please connect to sensors first.")

# Sensor Data page
elif page == "Sensor Data":
    st.header("Water Quality Sensor Data")
    
    # Upload sensor data
    with st.expander("Upload Sensor Data"):
        uploaded_file = st.file_uploader("Upload Sensor Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            df = process_uploaded_file(uploaded_file, "sensor")
            if df is not None:
                st.success("File uploaded successfully!")
                st.dataframe(df.head())
                
                if st.button("Add to Sensor Data"):
                    if st.session_state.sensor_data.empty:
                        st.session_state.sensor_data = df
                    else:
                        st.session_state.sensor_data = pd.concat([st.session_state.sensor_data, df], ignore_index=True)
                    st.success("Uploaded data added to sensor dataset!")
    
    # Input form for sensor data
    with st.expander("Add New Sensor Reading"):
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Location", list(st.session_state.village_coordinates.keys()) if st.session_state.village_coordinates else ["Village-1", "Village-2"])
            timestamp = st.date_input("Date", datetime.date.today())
            pH = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)
            turbidity = st.slider("Turbidity (NTU)", 0.0, 100.0, 5.0, 0.1)
            TDS = st.slider("TDS (ppm)", 0, 2000, 300, 10)
        with col2:
            residual_chlorine = st.slider("Residual Chlorine (mg/L)", 0.0, 5.0, 0.2, 0.1)
            temperature = st.slider("Temperature (°C)", 0.0, 40.0, 25.0, 0.1)
            rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 0.0, 1.0)
            
        if st.button("Save Sensor Data"):
            # Calculate simple risk score (in real implementation, use a more sophisticated formula)
            risk_score = (abs(pH-7)/7 * 25 + 
                         min(turbidity/100, 1) * 25 + 
                         min(TDS/2000, 1) * 25 + 
                         (1 - min(residual_chlorine/2, 1)) * 25)
            
            new_data = pd.DataFrame([{
                'timestamp': timestamp,
                'location': location,
                'pH': pH,
                'turbidity': turbidity,
                'TDS': TDS,
                'residual_chlorine': residual_chlorine,
                'temperature': temperature,
                'rainfall': rainfall,
                'risk_score': risk_score,
                'water_safety': 'safe' if risk_score < 30 else 'unsafe'
            }])
            
            if st.session_state.sensor_data.empty:
                st.session_state.sensor_data = new_data
            else:
                st.session_state.sensor_data = pd.concat([st.session_state.sensor_data, new_data], ignore_index=True)
            
            st.success("Sensor data saved successfully!")
    
    # Display existing data with delete option
    if not st.session_state.sensor_data.empty:
        st.subheader("Current Sensor Data")
        
        # Show data size and option to delete all
        st.write(f"Total records: {len(st.session_state.sensor_data)}")
        if st.button("Delete All Sensor Data"):
            st.session_state.sensor_data = pd.DataFrame()
            st.success("All sensor data deleted!")
        
        st.dataframe(st.session_state.sensor_data)
        
        # Basic visualization
        col1, col2 = st.columns(2)
        with col1:
            st.write("Risk Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            st.session_state.sensor_data['risk_score'].hist(bins=20, ax=ax)
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        
        with col2:
            st.write("Water Safety Status")
            safety_counts = st.session_state.sensor_data['water_safety'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(safety_counts, labels=safety_counts.index, autopct='%1.1f%%')
            ax.set_title("Water Safety Distribution")
            st.pyplot(fig)
    else:
        st.info("No sensor data available. Add data using the form above.")

# Symptom Analysis page
elif page == "Symptom Analysis":
    st.header("Symptom Analysis")
    
    # Upload symptom data
    with st.expander("Upload Symptom Data"):
        uploaded_file = st.file_uploader("Upload Symptom Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            df = process_uploaded_file(uploaded_file, "symptom")
            if df is not None:
                st.success("File uploaded successfully!")
                st.dataframe(df.head())
                
                if st.button("Add to Symptom Data"):
                    if st.session_state.symptom_data.empty:
                        st.session_state.symptom_data = df
                    else:
                        st.session_state.symptom_data = pd.concat([st.session_state.symptom_data, df], ignore_index=True)
                    st.success("Uploaded data added to symptom dataset!")
    
    # Input for symptom reporting
    with st.expander("Report Symptoms"):
        location = st.selectbox("Location", list(st.session_state.village_coordinates.keys()) if st.session_state.village_coordinates else ["Village-1", "Village-2"])
        report_date = st.date_input("Report Date", datetime.date.today())
        symptoms_text = st.text_area("Describe symptoms observed in the community:", height=150)
        
        if st.button("Analyze Symptoms"):
            if symptoms_text:
                risk_scores = process_symptoms(symptoms_text)
                
                new_data = pd.DataFrame([{
                    'timestamp': report_date,
                    'location': location,
                    'symptoms_text': symptoms_text,
                    'cholera_risk': risk_scores['cholera_risk'],
                    'typhoid_risk': risk_scores['typhoid_risk'],
                    'dysentery_risk': risk_scores['dysentery_risk']
                }])
                
                if st.session_state.symptom_data.empty:
                    st.session_state.symptom_data = new_data
                else:
                    st.session_state.symptom_data = pd.concat([st.session_state.symptom_data, new_data], ignore_index=True)
                
                st.success("Symptoms analyzed successfully!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                col1.metric("Cholera Risk", f"{risk_scores['cholera_risk']:.1f}%")
                col2.metric("Typhoid Risk", f"{risk_scores['typhoid_risk']:.1f}%")
                col3.metric("Dysentery Risk", f"{risk_scores['dysentery_risk']:.1f}%")
            else:
                st.warning("Please enter symptoms to analyze.")
    
    # Display symptom data with delete option
    if not st.session_state.symptom_data.empty:
        st.subheader("Symptom Analysis History")
        
        # Show data size and option to delete all
        st.write(f"Total records: {len(st.session_state.symptom_data)}")
        if st.button("Delete All Symptom Data"):
            st.session_state.symptom_data = pd.DataFrame()
            st.success("All symptom data deleted!")
        
        st.dataframe(st.session_state.symptom_data)
        
        # Visualize disease risks over time
        if 'timestamp' in st.session_state.symptom_data.columns:
            st.subheader("Disease Risk Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            for disease in ['cholera_risk', 'typhoid_risk', 'dysentery_risk']:
                if disease in st.session_state.symptom_data.columns:
                    ax.plot(st.session_state.symptom_data['timestamp'], 
                           st.session_state.symptom_data[disease], 
                           label=disease.replace('_risk', '').title())
            
            ax.set_xlabel("Date")
            ax.set_ylabel("Risk Score (%)")
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

# Weather Data page
elif page == "Weather Data":
    st.header("Weather Data Input")
    
    # Upload weather data
    with st.expander("Upload Weather Data"):
        uploaded_file = st.file_uploader("Upload Weather Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            df = process_uploaded_file(uploaded_file, "weather")
            if df is not None:
                st.session_state.weather_excel_data = df
                st.success("Weather data uploaded successfully!")
                st.dataframe(df.head())
    
    # Input for weather data
    with st.expander("Add Weather Data"):
        location = st.selectbox("Location", list(st.session_state.village_coordinates.keys()) if st.session_state.village_coordinates else ["Village-1", "Village-2"])
        date = st.date_input("Date", datetime.date.today())
        
        if st.button("Fetch Weather Data"):
            weather_data = get_weather_data(location, date)
            
            new_data = pd.DataFrame([{
                'timestamp': date,
                'location': location,
                'rainfall': weather_data['rainfall'],
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity']
            }])
            
            if st.session_state.weather_data.empty:
                st.session_state.weather_data = new_data
            else:
                st.session_state.weather_data = pd.concat([st.session_state.weather_data, new_data], ignore_index=True)
            
            st.success("Weather data fetched successfully!")
    
    # Display weather data with delete option
    if not st.session_state.weather_data.empty:
        st.subheader("Weather Data History")
        
        # Show data size and option to delete all
        st.write(f"Total records: {len(st.session_state.weather_data)}")
        if st.button("Delete All Weather Data"):
            st.session_state.weather_data = pd.DataFrame()
            st.success("All weather data deleted!")
        
        st.dataframe(st.session_state.weather_data)
        
        # Visualize rainfall data
        if 'timestamp' in st.session_state.weather_data.columns:
            st.subheader("Rainfall Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(st.session_state.weather_data['timestamp'], 
                  st.session_state.weather_data['rainfall'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Rainfall (mm)")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

# Sanitation Survey page
elif page == "Sanitation Survey":
    st.header("Sanitation Survey")
    
    # Upload survey data
    with st.expander("Upload Survey Data"):
        uploaded_file = st.file_uploader("Upload Survey Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            df = process_uploaded_file(uploaded_file, "survey")
            if df is not None:
                st.success("File uploaded successfully!")
                st.dataframe(df.head())
                
                if st.button("Add to Survey Data"):
                    if st.session_state.survey_data.empty:
                        st.session_state.survey_data = df
                    else:
                        st.session_state.survey_data = pd.concat([st.session_state.survey_data, df], ignore_index=True)
                    st.success("Uploaded data added to survey dataset!")
    
    # Input form for sanitation survey
    with st.expander("Conduct Sanitation Survey"):
        location = st.selectbox("Location", list(st.session_state.village_coordinates.keys()) if st.session_state.village_coordinates else ["Village-1", "Village-2"])
        survey_date = st.date_input("Survey Date", datetime.date.today())
        
        col1, col2 = st.columns(2)
        with col1:
            water_source = st.selectbox("Primary Water Source", 
                                      ["Tap", "Well", "Spring", "River", "Pond", "Other"])
            water_treatment = st.selectbox("Water Treatment Method", 
                                         ["None", "Boiling", "Chlorination", "Filter", "Other"])
            sanitation_facilities = st.slider("Households with Sanitation Facilities (%)", 0, 100, 50)
        
        with col2:
            open_defecation = st.slider("Open Defecation Practice (%)", 0, 100, 20)
            waste_management = st.slider("Waste Management Effectiveness (1-10)", 1, 10, 5)
        
        if st.button("Submit Survey"):
            # Calculate sanitation score
            sanitation_score = (sanitation_facilities * 0.3 + 
                              (100 - open_defecation) * 0.3 + 
                              waste_management * 10 * 0.4)
            
            new_data = pd.DataFrame([{
                'timestamp': survey_date,
                'location': location,
                'water_source_type': water_source,
                'water_treatment': water_treatment,
                'sanitation_facilities': sanitation_facilities,
                'open_defecation': open_defecation,
                'waste_management': waste_management,
                'sanitation_score': sanitation_score
            }])
            
            if st.session_state.survey_data.empty:
                st.session_state.survey_data = new_data
            else:
                st.session_state.survey_data = pd.concat([st.session_state.survey_data, new_data], ignore_index=True)
            
            st.success("Sanitation survey submitted successfully!")
    
    # Display survey data with delete option
    if not st.session_state.survey_data.empty:
        st.subheader("Sanitation Survey History")
        
        # Show data size and option to delete all
        st.write(f"Total records: {len(st.session_state.survey_data)}")
        if st.button("Delete All Survey Data"):
            st.session_state.survey_data = pd.DataFrame()
            st.success("All survey data deleted!")
        
        st.dataframe(st.session_state.survey_data)
        
        # Visualize sanitation scores
        st.subheader("Sanitation Scores by Location")
        fig, ax = plt.subplots(figsize=(10, 6))
        locations = st.session_state.survey_data['location'].unique()
        scores = [st.session_state.survey_data[st.session_state.survey_data['location'] == loc]['sanitation_score'].mean() 
                 for loc in locations]
        
        ax.bar(locations, scores)
        ax.set_xlabel("Location")
        ax.set_ylabel("Sanitation Score")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

# Village Coordinates page
elif page == "Village Coordinates":
    st.header("Village Coordinates Management")
    
    # Input for village coordinates
    with st.expander("Add Village Coordinates"):
        village_name = st.text_input("Village Name")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=25.0, format="%.6f")
        with col2:
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=91.0, format="%.6f")
        
        if st.button("Add Village"):
            if village_name:
                st.session_state.village_coordinates[village_name] = (latitude, longitude)
                st.success(f"Added {village_name} with coordinates ({latitude}, {longitude})")
            else:
                st.warning("Please enter a village name.")
    
    # Display village coordinates with delete option
    if st.session_state.village_coordinates:
        st.subheader("Current Village Coordinates")
        
        # Show data size and option to delete all
        st.write(f"Total villages: {len(st.session_state.village_coordinates)}")
        if st.button("Delete All Village Data"):
            st.session_state.village_coordinates = {}
            st.success("All village data deleted!")
        
        villages_df = pd.DataFrame.from_dict(st.session_state.village_coordinates, 
                                           orient='index', 
                                           columns=['Latitude', 'Longitude'])
        st.dataframe(villages_df)
        
        # Display on map
        st.subheader("Village Locations")
        if not villages_df.empty:
            # Create a map centered on the first village
            first_village = list(st.session_state.village_coordinates.keys())[0]
            map_center = st.session_state.village_coordinates[first_village]
            
            m = folium.Map(location=map_center, zoom_start=7)
            
            # Add markers for each village
            for village, coords in st.session_state.village_coordinates.items():
                folium.Marker(
                    coords,
                    popup=village,
                    tooltip=village
                ).add_to(m)
            
            # Display the map
            folium_static(m, width=1000, height=600)

# Fusion Analysis page
elif page == "Fusion Analysis":
    st.header("Fusion Analysis")
    
    # Check if we have data from all sources
    has_sensor = not st.session_state.sensor_data.empty
    has_symptoms = not st.session_state.symptom_data.empty
    has_weather = not st.session_state.weather_data.empty
    has_survey = not st.session_state.survey_data.empty
    
    if not (has_sensor and has_symptoms and has_weather and has_survey):
        st.warning("Please add data from all sources (sensor, symptoms, weather, survey) to perform fusion analysis.")
    else:
        # Merge data from different sources
        # This is a simplified example - in a real application, you would need more sophisticated data fusion
        merged_data = st.session_state.sensor_data.copy()
        
        # Add symptom data
        if has_symptoms:
            symptom_agg = st.session_state.symptom_data.groupby(['location', 'timestamp']).agg({
                'cholera_risk': 'mean',
                'typhoid_risk': 'mean',
                'dysentery_risk': 'mean'
            }).reset_index()
            
            merged_data = pd.merge(merged_data, symptom_agg, on=['location', 'timestamp'], how='left')
        
        # Add weather data
        if has_weather:
            weather_agg = st.session_state.weather_data.groupby(['location', 'timestamp']).agg({
                'rainfall': 'mean',
                'temperature': 'mean',
                'humidity': 'mean'
            }).reset_index()
            
            merged_data = pd.merge(merged_data, weather_agg, on=['location', 'timestamp'], how='left')
        
        # Add survey data
        if has_survey:
            survey_agg = st.session_state.survey_data.groupby(['location', 'timestamp']).agg({
                'sanitation_score': 'mean'
            }).reset_index()
            
            merged_data = pd.merge(merged_data, survey_agg, on=['location', 'timestamp'], how='left')
        
        # Display merged data
        st.subheader("Fused Data from All Sources")
        st.dataframe(merged_data)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
        corr_matrix = merged_data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Correlation Matrix of All Variables")
        st.pyplot(fig)
        
        # Key insights
        st.subheader("Key Insights")
        if 'risk_score' in corr_matrix.columns:
            risk_correlations = corr_matrix['risk_score'].sort_values(key=abs, ascending=False)
            top_factors = risk_correlations.index[1:4]  # Exclude self-correlation
            
            st.write("Factors most correlated with risk score:")
            for factor in top_factors:
                corr_value = risk_correlations[factor]
                st.write(f"- {factor}: {corr_value:.3f}")

# Predictive Analytics page
elif page == "Predictive Analytics":
    st.header("Predictive Analytics")
    
    # Check if we have enough data for training
    if st.session_state.sensor_data.empty:
        st.warning("Please add sensor data first to train models.")
    else:
        # Prepare data for machine learning
        df = st.session_state.sensor_data.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        if 'water_safety' in df.columns:
            df['water_safety_encoded'] = le.fit_transform(df['water_safety'])
        
        # Select features and target
        features = ['pH', 'turbidity', 'TDS', 'residual_chlorine', 'temperature', 'rainfall']
        target = 'water_safety_encoded'
        
        if target in df.columns and all(feat in df.columns for feat in features):
            # Remove rows with missing values
            ml_data = df[features + [target]].dropna()
            
            if len(ml_data) > 10:  # Ensure we have enough data
                X = ml_data[features]
                y = ml_data[target]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train multiple models and select the best one
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
                }
                
                best_model = None
                best_accuracy = 0
                model_results = {}
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_results[name] = {
                        'model': model,
                        'accuracy': accuracy
                    }
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                
                st.session_state.trained_model = best_model
                st.session_state.model_accuracy = best_accuracy
                
                # Display results
                st.subheader("Model Performance Comparison")
                results_df = pd.DataFrame.from_dict(
                    {k: v['accuracy'] for k, v in model_results.items()}, 
                    orient='index', 
                    columns=['Accuracy']
                )
                st.dataframe(results_df.style.format({'Accuracy': '{:.2%}'}))
                
                # Feature importance for tree-based models
                if hasattr(best_model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'feature': features,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['feature'], importance_df['importance'])
                    ax.set_xlabel("Importance")
                    ax.set_title("Feature Importance for Water Safety Prediction")
                    st.pyplot(fig)
                
                # Prediction interface
                st.subheader("Make Prediction")
                col1, col2 = st.columns(2)
                with col1:
                    pred_pH = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1, key='pred_ph')
                    pred_turbidity = st.slider("Turbidity (NTU)", 0.0, 100.0, 5.0, 0.1, key='pred_turb')
                    pred_TDS = st.slider("TDS (ppm)", 0, 2000, 300, 10, key='pred_tds')
                
                with col2:
                    pred_residual_chlorine = st.slider("Residual Chlorine (mg/L)", 0.0, 5.0, 0.2, 0.1, key='pred_cl')
                    pred_temperature = st.slider("Temperature (°C)", 0.0, 40.0, 25.0, 0.1, key='pred_temp')
                    pred_rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 0.0, 1.0, key='pred_rain')
                
                if st.button("Predict Water Safety"):
                    input_data = [[pred_pH, pred_turbidity, pred_TDS, pred_residual_chlorine, pred_temperature, pred_rainfall]]
                    prediction = best_model.predict(input_data)[0]
                    prediction_proba = best_model.predict_proba(input_data)[0]
                    
                    safety_status = "Safe" if prediction == 1 else "Unsafe"
                    safe_confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
                    
                    st.success(f"Predicted Water Safety: {safety_status} ({safe_confidence:.2%} confidence)")
                    
                    # Show detailed probabilities
                    st.write("Detailed probabilities:")
                    prob_df = pd.DataFrame({
                        'Class': ['Unsafe', 'Safe'],
                        'Probability': prediction_proba
                    })
                    st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
            else:
                st.warning("Not enough data for training. Please add more sensor readings.")
        else:
            st.warning("Required features not available in the data.")

# Risk Dashboard page
elif page == "Risk Dashboard":
    st.header("Risk Dashboard")
    
    # Check if we have data to display
    if st.session_state.sensor_data.empty and st.session_state.symptom_data.empty:
        st.warning("Please add data to view the risk dashboard.")
    else:
        # Create a comprehensive risk overview
        col1, col2, col3, col4 = st.columns(4)
        
        # Overall risk metrics
        if not st.session_state.sensor_data.empty:
            avg_risk = st.session_state.sensor_data['risk_score'].mean()
            high_risk_locations = len(st.session_state.sensor_data[st.session_state.sensor_data['risk_score'] > 50])
            total_locations = st.session_state.sensor_data['location'].nunique()
            
            with col1:
                st.metric("Average Risk Score", f"{avg_risk:.1f}")
            with col2:
                st.metric("High Risk Locations", f"{high_risk_locations}/{total_locations}")
        
        # Disease risk metrics
        if not st.session_state.symptom_data.empty:
            avg_cholera = st.session_state.symptom_data['cholera_risk'].mean()
            avg_typhoid = st.session_state.symptom_data['typhoid_risk'].mean()
            
            with col3:
                st.metric("Avg. Cholera Risk", f"{avg_cholera:.1f}%")
            with col4:
                st.metric("Avg. Typhoid Risk", f"{avg_typhoid:.1f}%")
        
        # Risk trends over time
        if not st.session_state.sensor_data.empty and 'timestamp' in st.session_state.sensor_data.columns:
            st.subheader("Risk Trends Over Time")
            
            # Aggregate risk by time
            time_risk = st.session_state.sensor_data.groupby('timestamp')['risk_score'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_risk['timestamp'], time_risk['risk_score'], marker='o')
            ax.set_xlabel("Date")
            ax.set_ylabel("Average Risk Score")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Location-based risk heatmap
        if not st.session_state.sensor_data.empty and st.session_state.village_coordinates:
            st.subheader("Risk Map by Location")
            
            # Calculate average risk per location
            location_risk = st.session_state.sensor_data.groupby('location')['risk_score'].mean().reset_index()
            
            # Create a map
            if not location_risk.empty:
                # Find center of all villages
                avg_lat = np.mean([coords[0] for coords in st.session_state.village_coordinates.values()])
                avg_lon = np.mean([coords[1] for coords in st.session_state.village_coordinates.values()])
                
                m = folium.Map(location=[avg_lat, avg_lon], zoom_start=7)
                
                # Add markers with risk information
                for _, row in location_risk.iterrows():
                    location = row['location']
                    risk = row['risk_score']
                    
                    if location in st.session_state.village_coordinates:
                        lat, lon = st.session_state.village_coordinates[location]
                        
                        # Color code by risk level
                        color = 'green' if risk < 30 else 'orange' if risk < 70 else 'red'
                        
                        folium.CircleMarker(
                            [lat, lon],
                            radius=10,
                            popup=f"{location}: Risk Score {risk:.1f}",
                            tooltip=location,
                            color=color,
                            fill=True,
                            fillColor=color
                        ).add_to(m)
                
                folium_static(m, width=1000, height=600)

# Regional Analysis page
elif page == "Regional Analysis":
    st.header("Regional Analysis")
    
    # Check if we have data and coordinates
    if st.session_state.sensor_data.empty or not st.session_state.village_coordinates:
        st.warning("Please add sensor data and village coordinates to perform regional analysis.")
    else:
        # Merge sensor data with coordinates
        location_risk = st.session_state.sensor_data.groupby('location')['risk_score'].mean().reset_index()
        
        # Create a DataFrame with coordinates
        coords_df = pd.DataFrame.from_dict(st.session_state.village_coordinates, 
                                         orient='index', 
                                         columns=['latitude', 'longitude'])
        coords_df.index.name = 'location'
        coords_df = coords_df.reset_index()
        
        # Merge with risk data
        regional_data = pd.merge(location_risk, coords_df, on='location', how='inner')
        
        if not regional_data.empty:
            # Display regional statistics
            st.subheader("Regional Risk Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Risk Score Distribution by Region")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(regional_data['location'], regional_data['risk_score'])
                ax.set_xlabel("Location")
                ax.set_ylabel("Average Risk Score")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            
            with col2:
                st.write("Risk Categories")
                risk_categories = pd.cut(regional_data['risk_score'], 
                                       bins=[0, 30, 70, 100], 
                                       labels=['Low', 'Medium', 'High'])
                category_counts = risk_categories.value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
                ax.set_title("Risk Category Distribution")
                st.pyplot(fig)
            
            # Spatial analysis
            st.subheader("Spatial Risk Analysis")
            
            # Calculate distance matrix (simplified)
            st.write("Distance-Based Risk Correlation")
            distances = []
            risk_diffs = []
            
            for i in range(len(regional_data)):
                for j in range(i+1, len(regional_data)):
                    loc1 = regional_data.iloc[i]
                    loc2 = regional_data.iloc[j]
                    
                    # Calculate distance (simplified)
                    distance = calculate_distance((loc1['latitude'], loc1['longitude']), 
                                                (loc2['latitude'], loc2['longitude']))
                    risk_diff = abs(loc1['risk_score'] - loc2['risk_score'])
                    
                    distances.append(distance)
                    risk_diffs.append(risk_diff)
            
            # Plot distance vs risk difference
            if distances and risk_diffs:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(distances, risk_diffs, alpha=0.6)
                ax.set_xlabel("Distance between villages (km)")
                ax.set_ylabel("Difference in risk score")
                ax.set_title("Distance vs Risk Difference")
                st.pyplot(fig)

# Footer with larger font
st.sidebar.divider()
st.sidebar.markdown("""
<div style="font-size: 16px;">
For questions or support, please contact the development team.<br>
This is a prototype for demonstration purposes.
</div>
""", unsafe_allow_html=True)