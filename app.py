from generate_water_dataset import generate_water_data
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import datetime
import io
import re
from collections import defaultdict
import sqlite3
import os

# Set page configuration with wider layout
st.set_page_config(
    page_title="Water-Borne Disease Prediction",
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
            pH REAL,
            turbidity REAL,
            TDS REAL,
            cl REAL,
            dop REAL,
            temp REAL,
            risk_score REAL,
            cholera INTEGER,
            dengue INTEGER,
            pneumonia INTEGER,
            typhoid INTEGER,
            jaundice INTEGER,
            probable_disease TEXT,
            probability REAL,
            location TEXT,
            num_water_sources INTEGER,
            water_safety TEXT,
            UNIQUE(timestamp, location)  -- Prevent duplicate entries
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Title and description with larger font
st.markdown("""
<style>
.title-box {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #0ea5e9;
    font-family: 'Segoe UI', sans-serif;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #334155;
    margin-bottom: 25px;
}
</style>

<div class="title-box">💧 Water-Borne Disease Prediction System</div>
<div class="subtitle">An AI-powered system for sensor-based, symptom-based, and fusion analysis of water safety</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.description-box {
    background-color: #f9fafb;
    font-size: 18px;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
    color: #374151;
    # text-align: center;
    line-height: 1.6;
    margin-bottom: 25px;
    font-family: 'Segoe UI', sans-serif;
}
.description-box ol {
    list-style-position: inside;
    padding-left: 0;
    margin: 8px 0 0 0;
}
</style>

<div class="description-box">
This system predicts water-borne diseases in rural areas of East India using:
<ol>
<li><b>Sensor data analysis</b> (pH, turbidity, TDS, etc.)</li>
<li><b>Symptom analysis</b> through natural language processing</li>
<li><b>Fusion of both approaches</b> for enhanced accuracy</li>
</ol>
</div>
""", unsafe_allow_html=True)


# Initialize session state for data storage
if 'continuous_learning_data' not in st.session_state:
    st.session_state.continuous_learning_data = pd.DataFrame()
if 'kaggle_df' not in st.session_state:
    st.session_state.kaggle_df = None

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
        df = generate_water_data()  # This function should return the DataFrame
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Save the generated dataset
        os.makedirs("Generate_data_set", exist_ok=True)
        df.to_csv(file_path, index=False)
        st.success("Dataset generated and saved successfully!")
        return df

# Symptom keyword mapping
symptom_keywords = {
    'cholera': ['vomiting', 'diarrhea', 'dehydration', 'rice water stools', 'leg cramps'],
    'dengue': ['fever', 'headache', 'muscle pain', 'joint pain', 'rash', 'bleeding'],
    'pneumonia': ['cough', 'fever', 'difficulty breathing', 'chest pain', 'phlegm'],
    'typhoid': ['fever', 'headache', 'abdominal pain', 'constipation', 'diarrhea'],
    'jaundice': ['yellow skin', 'yellow eyes', 'dark urine', 'fatigue', 'abdominal pain']
}

# Sidebar for navigation with larger font
st.sidebar.markdown("<h2 style='font-size: 24px;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Home", "Sensor Analysis (M1)", "Symptom Analysis (M2)", "Fusion Analysis", "Dashboard", "Data Management"])

# Load sample data if no file is uploaded
@st.cache_data
def load_sample_data():
    # Create sample data based on the provided structure
    sample_data = {
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
        'pH': np.random.uniform(5.5, 9.0, 100),
        'turbidity': np.random.uniform(0.5, 15.0, 100),
        'TDS': np.random.uniform(200, 1200, 100),
        'cl': np.random.uniform(0.0, 0.8, 100),
        'dop': np.random.uniform(3.0, 7.0, 100),
        'temp': np.random.uniform(25.0, 35.0, 100),
        'risk_score': np.random.uniform(0.1, 0.95, 100),
        'cholera': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'dengue': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'pneumonia': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'typhoid': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'jaundice': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        'probable_disease': np.random.choice(['Cholera', 'Dengue', 'Pneumonia', 'Typhoid', 'Jaundice'], 100),
        'probability': np.random.uniform(0.8, 0.95, 100),
        'location': [f'tribe-{chr(i)}' for i in range(65, 91)] * 4,
        'num_water_sources': np.random.randint(1, 11, 100),
        'water_safety': np.random.choice(['Safe', 'Unsafe'], 100, p=[0.7, 0.3])
    }
    return pd.DataFrame(sample_data)

# Function to process natural language symptoms
def process_symptoms(text):
    symptoms = defaultdict(int)
    text = text.lower()
    
    for disease, keywords in symptom_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                symptoms[disease] = 1
                
    return symptoms

# Home page
if page == "Home":
    st.header("Welcome to the Water-Borne Disease Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Overview")
        st.markdown("""
        <div style="font-size: 16px;">
        This system uses three integrated approaches:
        
        1. Sensor Analysis (M1): 
           - Input: Water quality parameters (pH, turbidity, TDS, etc.)
           - Output: Risk score and Safe/Unsafe classification
        
        2. Symptom Analysis (M2):
           - Input: Natural language description of patient symptoms
           - Output: Tokenized symptoms and disease probabilities
    
        3. Fusion Analysis:
           - Combines both sensor and symptom data
           - Provides enhanced predictions
           - Updates dataset for continuous learning
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("How to Use")
        st.markdown("""
        <div style="font-size: 16px;">
                    
        1. Sensor Analysis: Use manual input, generated dataset, or upload Kaggle dataset            
        2. Symptom Analysis: Describe symptoms in natural language
        3. Fusion Analysis: Combine both data sources for enhanced prediction
        4. Dashboard: View analytics and trends
        5. Data Management: Manage collected data for continuous learning
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Key Features")
        st.markdown("""
        <div style="font-size: 16px;">
                    
        - Natural language processing for symptoms
        - Continuous learning capabilities
        - Tribe-specific analytics
        - Alert system for high-risk situations
        - Mobile-friendly interface
                    
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown("""
        <style>
        .info-box {
            background-color: #e0f2fe;
            color: #075985;
            border-left: 6px solid #0284c7;
            padding: 14px 18px;
            border-radius: 10px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px;
            line-height: 1.5;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.04);
            margin-top: 15px;
        }
        .info-box a {
            color: #0c4a6e;
            font-weight: bold;
            text-decoration: none;
        }
        .info-box a:hover {
            text-decoration: underline;
        }
        </style>

        <div class="info-box">
        💡 <b>Tip:</b> Navigate using the sidebar to explore the different features of this system.<br>
        🔗 <b>Source Code:</b> <a href="https://github.com/hi-it-isDebayan/Note" target="_blank">GitHub Repository</a>
        </div>
        """, unsafe_allow_html=True)

# Sensor Analysis (M1) page
elif page == "Sensor Analysis (M1)":
    st.header("Sensor Data Analysis (M1)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Manual Input", "Generated Dataset", "Kaggle Dataset", "View Data"])
    
    with tab1:
        st.subheader("Manual Sensor Data Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pH = st.slider("pH", 5.0, 9.0, 7.0, 0.1)
            turbidity = st.slider("Turbidity", 0.0, 20.0, 5.0, 0.1)
            TDS = st.slider("TDS", 0, 1500, 500, 10)
            
        with col2:
            cl = st.slider("Chloride (cl)", 0.0, 1.0, 0.3, 0.01)
            dop = st.slider("Dissolved Oxygen (dop)", 0.0, 10.0, 5.0, 0.1)
            temp = st.slider("Temperature", 20.0, 40.0, 30.0, 0.1)
            
        location = st.selectbox("Location", [f"tribe-{chr(i)}" for i in range(65, 91)])
        num_water_sources = st.number_input("Number of Water Sources", min_value=1, max_value=20, value=1)
        
        if st.button("Analyze Sensor Data", key="analyze_sensor"):
            # Calculate risk score (simplified for demonstration)
            sensor_values = [pH, turbidity, TDS, cl, dop, temp]
            weights = [0.2, 0.25, 0.15, 0.1, 0.2, 0.1]
            
            # Normalize values (simplified)
            normalized_values = [
                (pH - 5.0) / (9.0 - 5.0),
                turbidity / 20.0,
                TDS / 1500.0,
                cl / 1.0,
                dop / 10.0,
                (temp - 20.0) / (40.0 - 20.0)
            ]
            
            risk_score = sum([val * weight for val, weight in zip(normalized_values, weights)])
            water_safety = "Unsafe" if risk_score > 0.6 else "Safe"
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{risk_score:.3f}")
                
            with col2:
                st.metric("Water Safety", water_safety)
                
            with col3:
                if water_safety == "Unsafe":
                    st.error("⚠ Unsafe water detected!")
                else:
                    st.success("✅ Water is safe")
            
            # Store result for potential fusion analysis
            manual_result = {
                'timestamp': datetime.datetime.now().isoformat(),
                'location': location,
                'pH': pH,
                'turbidity': turbidity,
                'TDS': TDS,
                'cl': cl,
                'dop': dop,
                'temp': temp,
                'risk_score': risk_score,
                'water_safety': water_safety,
                'cholera': 0,
                'dengue': 0,
                'pneumonia': 0,
                'typhoid': 0,
                'jaundice': 0,
                'probable_disease': '',
                'probability': 0.0,
                'num_water_sources': num_water_sources
            }
            
            st.session_state.manual_sensor_result = manual_result
            
            # ✅ Save to DB with proper error handling
            try:
                conn = sqlite3.connect("water_data.db")
                cursor = conn.cursor()
                
                # Check if a similar entry already exists
                cursor.execute("""
                    SELECT COUNT(*) FROM sensor_data 
                    WHERE timestamp = ? AND location = ?
                """, (manual_result['timestamp'], manual_result['location']))
                
                count = cursor.fetchone()[0]
                
                if count == 0:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO sensor_data 
                        (timestamp, pH, turbidity, TDS, cl, dop, temp, risk_score, water_safety, 
                         cholera, dengue, pneumonia, typhoid, jaundice, probable_disease, 
                         probability, location, num_water_sources)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        manual_result['timestamp'],
                        manual_result['pH'],
                        manual_result['turbidity'],
                        manual_result['TDS'],
                        manual_result['cl'],
                        manual_result['dop'],
                        manual_result['temp'],
                        manual_result['risk_score'],
                        manual_result['water_safety'],
                        manual_result['cholera'],
                        manual_result['dengue'],
                        manual_result['pneumonia'],
                        manual_result['typhoid'],
                        manual_result['jaundice'],
                        manual_result['probable_disease'],
                        manual_result['probability'],
                        manual_result['location'],
                        manual_result['num_water_sources']
                    ))
                    
                    conn.commit()
                    st.success("Manual data saved to database!")
                else:
                    st.warning("Similar entry already exists in the database.")
                
                conn.close()
            except Exception as e:
                st.error(f"Error saving manual data: {e}")
            
            st.success("Analysis complete! You can now use this result in Fusion Analysis.")
    
    with tab2:
        st.subheader("Generated Dataset")
        
        if st.button("Load Generated Dataset", key="load_generated"):
            generated_df = load_generated_data()
            if generated_df is not None:
                st.session_state.kaggle_df = generated_df
                st.success("Generated dataset loaded successfully!")
            else:
                st.error("Failed to load generated dataset")
                
        if st.session_state.kaggle_df is not None and isinstance(st.session_state.kaggle_df, pd.DataFrame):
            st.write("Generated Dataset Preview:")
            st.dataframe(st.session_state.kaggle_df.head())
            
            # Check if we have the required columns
            sensor_features = ['pH', 'turbidity', 'TDS', 'cl', 'dop', 'temp']
            if all(feature in st.session_state.kaggle_df.columns for feature in sensor_features):
                if 'risk_score' not in st.session_state.kaggle_df.columns:
                    # Create a simple risk score calculation for demonstration
                    st.info("Calculating risk scores...")
                    # Normalize each parameter and create a weighted risk score
                    df_normalized = st.session_state.kaggle_df[sensor_features].copy()
                    for col in sensor_features:
                        df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())
                    
                    # Define weights for each parameter (these would be learned in a real model)
                    weights = {
                        'pH': 0.2,
                        'turbidity': 0.25,
                        'TDS': 0.15,
                        'cl': 0.1,
                        'dop': 0.2,
                        'temp': 0.1
                    }
                    
                    st.session_state.kaggle_df['risk_score'] = 0
                    for col, weight in weights.items():
                        st.session_state.kaggle_df['risk_score'] += df_normalized[col] * weight
                    
                    # Classify as Safe/Unsafe
                    st.session_state.kaggle_df['water_safety'] = st.session_state.kaggle_df['risk_score'].apply(lambda x: 'Unsafe' if x > 0.6 else 'Safe')
                
                # ✅ Save to SQLite DB with proper handling
                try:
                    conn = sqlite3.connect("water_data.db")
                    cursor = conn.cursor()
                    
                    # Get existing timestamps and locations to avoid duplicates
                    cursor.execute("SELECT timestamp, location FROM sensor_data")
                    existing_records = cursor.fetchall()
                    existing_set = set((str(record[0]), record[1]) for record in existing_records)
                    
                    # Filter out duplicates
                    new_records = []
                    for _, row in st.session_state.kaggle_df.iterrows():
                        key = (str(row['timestamp']), row['location'])
                        if key not in existing_set:
                            new_records.append(row)
                    
                    if new_records:
                        new_df = pd.DataFrame(new_records)
                        new_df.to_sql("sensor_data", conn, if_exists="append", index=False)
                        st.success(f"Added {len(new_df)} new records to database!")
                    else:
                        st.info("No new records to add. All data already exists in the database.")
                    
                    conn.close()
                except Exception as e:
                    st.error(f"Error saving to database: {e}")
                    
                # Display results
                st.subheader("Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Records", len(st.session_state.kaggle_df))
                    unsafe_count = (st.session_state.kaggle_df['water_safety'] == 'Unsafe').sum()
                    st.metric("Unsafe Water Sources", f"{unsafe_count} ({unsafe_count/len(st.session_state.kaggle_df)*100:.1f}%)")
                
                with col2:
                    avg_risk = st.session_state.kaggle_df['risk_score'].mean()
                    st.metric("Average Risk Score", f"{avg_risk:.3f}")
                    if avg_risk > 0.6:
                        st.error("High average risk detected!")
                    
                # Show data with risk scores
                st.dataframe(st.session_state.kaggle_df[['timestamp', 'location'] + sensor_features + ['risk_score', 'water_safety']].head(10))
                
                # Download results with unique key
                csv = st.session_state.kaggle_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="sensor_analysis_results.csv",
                    mime="text/csv",
                    key="download_generated_data"  # Unique key added
                )
                
            else:
                missing = [f for f in sensor_features if f not in st.session_state.kaggle_df.columns]
                st.error(f"Missing sensor features: {', '.join(missing)}")
        else:
            st.info("Click the button to load the generated dataset.")
    
    with tab3:
        st.subheader("Upload Kaggle Dataset")
        
        # File uploader for Kaggle dataset
        uploaded_file = st.file_uploader("Choose a CSV file from Kaggle", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                kaggle_df = pd.read_csv(uploaded_file)
                st.session_state.kaggle_df = kaggle_df
                st.success("Kaggle dataset uploaded successfully!")
                
                st.write("Kaggle Dataset Preview:")
                st.dataframe(st.session_state.kaggle_df.head())
                
                # Check if we have the required columns
                sensor_features = ['pH', 'turbidity', 'TDS', 'cl', 'dop', 'temp']
                if all(feature in st.session_state.kaggle_df.columns for feature in sensor_features):
                    if 'risk_score' not in st.session_state.kaggle_df.columns:
                        # Create a simple risk score calculation for demonstration
                        st.info("Calculating risk scores...")
                        # Normalize each parameter and create a weighted risk score
                        df_normalized = st.session_state.kaggle_df[sensor_features].copy()
                        for col in sensor_features:
                            df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())
                        
                        # Define weights for each parameter (these would be learned in a real model)
                        weights = {
                            'pH': 0.2,
                            'turbidity': 0.25,
                            'TDS': 0.15,
                            'cl': 0.1,
                            'dop': 0.2,
                            'temp': 0.1
                        }
                        
                        st.session_state.kaggle_df['risk_score'] = 0
                        for col, weight in weights.items():
                            st.session_state.kaggle_df['risk_score'] += df_normalized[col] * weight
                        
                        # Classify as Safe/Unsafe
                        st.session_state.kaggle_df['water_safety'] = st.session_state.kaggle_df['risk_score'].apply(lambda x: 'Unsafe' if x > 0.6 else 'Safe')
                    
                    # ✅ Save to SQLite DB with proper handling
                    try:
                        conn = sqlite3.connect("water_data.db")
                        cursor = conn.cursor()
                        
                        # Get existing timestamps and locations to avoid duplicates
                        cursor.execute("SELECT timestamp, location FROM sensor_data")
                        existing_records = cursor.fetchall()
                        existing_set = set((str(record[0]), record[1]) for record in existing_records)
                        
                        # Filter out duplicates
                        new_records = []
                        for _, row in st.session_state.kaggle_df.iterrows():
                            key = (str(row['timestamp']), row['location'])
                            if key not in existing_set:
                                new_records.append(row)
                        
                        if new_records:
                            new_df = pd.DataFrame(new_records)
                            new_df.to_sql("sensor_data", conn, if_exists="append", index=False)
                            st.success(f"Added {len(new_df)} new records to database!")
                        else:
                            st.info("No new records to add. All data already exists in the database.")
                        
                        conn.close()
                    except Exception as e:
                        st.error(f"Error saving to database: {e}")
                        
                    # Display results
                    st.subheader("Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Records", len(st.session_state.kaggle_df))
                        unsafe_count = (st.session_state.kaggle_df['water_safety'] == 'Unsafe').sum()
                        st.metric("Unsafe Water Sources", f"{unsafe_count} ({unsafe_count/len(st.session_state.kaggle_df)*100:.1f}%)")
                    
                    with col2:
                        avg_risk = st.session_state.kaggle_df['risk_score'].mean()
                        st.metric("Average Risk Score", f"{avg_risk:.3f}")
                        if avg_risk > 0.6:
                            st.error("High average risk detected!")
                        
                    # Show data with risk scores
                    st.dataframe(st.session_state.kaggle_df[['timestamp', 'location'] + sensor_features + ['risk_score', 'water_safety']].head(10))
                    
                    # Download results with unique key
                    csv = st.session_state.kaggle_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sensor_analysis_results.csv",
                        mime="text/csv",
                        key="download_kaggle_data"  # Unique key added
                    )
                    
                else:
                    missing = [f for f in sensor_features if f not in st.session_state.kaggle_df.columns]
                    st.error(f"Missing sensor features: {', '.join(missing)}")
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.info("Please upload a CSV file from Kaggle.")
    
    with tab4:
        st.subheader("View Current Data")
        try:
            conn = sqlite3.connect("water_data.db")
            db_df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
            conn.close()
            
            # Fix timestamp parsing
            if not db_df.empty and 'timestamp' in db_df.columns:
                # Handle both ISO format and other formats
                try:
                    db_df['timestamp'] = pd.to_datetime(db_df['timestamp'], utc=True)
                except:
                    # If ISO format fails, try parsing without timezone
                    db_df['timestamp'] = pd.to_datetime(db_df['timestamp'], errors='coerce')
            
            if not db_df.empty:
                st.dataframe(db_df)
                st.metric("Total Records", len(db_df))
                
                # Add Clear Database button in View Data section
                if st.button("Clear Whole Database", key="clear_db_view_data"):
                    try:
                        conn = sqlite3.connect("water_data.db")
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM sensor_data")
                        conn.commit()
                        conn.close()
                        st.success("Database cleared successfully!")
                        # Use JavaScript to reload the page
                        st.markdown("<meta http-equiv='refresh' content='0'>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error clearing database: {e}")
            else:
                st.info("No data available yet. Please add data through manual input or dataset upload.")
        except Exception as e:
            st.error(f"Error reading from database: {e}")

# Symptom Analysis (M2) page
elif page == "Symptom Analysis (M2)":
    st.header("Symptom Analysis (M2)")
    
    st.info("Describe the patient's symptoms in natural language. The system will automatically identify relevant symptoms.")
    
    symptom_text = st.text_area(
        "Enter symptom description:",
        placeholder="Example: The patient has been vomiting and has a fever with muscle pain...",
        height=150
    )
    
    if st.button("Analyze Symptoms", key="analyze_symptoms"):
        if symptom_text:
            # Process the natural language symptoms
            symptoms = process_symptoms(symptom_text)
            
            # Display detected symptoms
            st.subheader("Detected Symptoms")
            symptom_cols = st.columns(5)
            
            for i, (disease, present) in enumerate(symptoms.items()):
                with symptom_cols[i % 5]:
                    if present:
                        st.error(f"{disease.capitalize()}** ✅")
                    else:
                        st.info(f"{disease.capitalize()}** ❌")
            
            # Calculate disease probabilities (simplified for demonstration)
            st.subheader("Disease Probability Estimates")
            
            # In a real system, this would be a trained model
            base_probabilities = {
                'cholera': 0.3,
                'dengue': 0.4,
                'pneumonia': 0.2,
                'typhoid': 0.1,
                'jaundice': 0.05
            }
            
            # Adjust probabilities based on symptoms
            for disease, present in symptoms.items():
                if present:
                    base_probabilities[disease] = min(base_probabilities[disease] * 2, 0.95)
            
            # Normalize probabilities
            total = sum(base_probabilities.values())
            disease_probabilities = {k: v/total for k, v in base_probabilities.items()}
            
            # Display probabilities
            prob_cols = st.columns(5)
            for i, (disease, prob) in enumerate(disease_probabilities.items()):
                with prob_cols[i % 5]:
                    st.metric(f"{disease.capitalize()} Probability", f"{prob:.2%}")
                    if prob > 0.5:
                        st.error("*High probability!*")
                    elif prob > 0.3:
                        st.warning("*Moderate probability*")
            
            # Store results for fusion analysis
            st.session_state.symptom_analysis_result = {
                'timestamp': datetime.datetime.now(),
                'symptom_text': symptom_text,
                'symptoms_detected': dict(symptoms),
                'disease_probabilities': disease_probabilities,
                'primary_disease': max(disease_probabilities, key=disease_probabilities.get)
            }
            
            st.success("Symptom analysis complete! You can now use this result in Fusion Analysis.")
        else:
            st.warning("Please enter symptom description.")

# Fusion Analysis page
elif page == "Fusion Analysis":
    st.header("Fusion Analysis")
    
    st.info("Combine sensor data and symptom analysis for enhanced prediction accuracy.")
    
    # Check if we have sensor data from either manual input or Kaggle
    try:
        conn = sqlite3.connect("water_data.db")
        db_df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
        conn.close()
        
        # Fix timestamp parsing
        if not db_df.empty and 'timestamp' in db_df.columns:
            try:
                db_df['timestamp'] = pd.to_datetime(db_df['timestamp'], utc=True)
            except:
                # If ISO format fails, try parsing without timezone
                db_df['timestamp'] = pd.to_datetime(db_df['timestamp'], errors='coerce')
        
        if db_df.empty:
            st.warning("Please perform sensor analysis first.")
            st.stop()
    except:
        st.warning("Please perform sensor analysis first.")
        st.stop()
        
    if 'symptom_analysis_result' not in st.session_state:
        st.warning("Please perform symptom analysis first.")
        st.stop()
    
    # Get the latest sensor result
    latest_sensor_data = db_df.iloc[-1].to_dict()
    
    st.subheader("Latest Sensor Data")
    st.json({k: v for k, v in latest_sensor_data.items() if not pd.isna(v)})
    
    # Get symptom results
    symptom_results = st.session_state.symptom_analysis_result
    st.subheader("Symptom Analysis Results")
    st.json(symptom_results)
    
    if st.button("Perform Fusion Analysis", key="perform_fusion"):
        with st.spinner("Performing fusion analysis..."):
            # Simulate fusion analysis (in a real system, this would be a trained model)
            # For demonstration, we'll create a weighted combination
            
            # Get the risk score from sensor analysis
            sensor_risk = latest_sensor_data['risk_score']
            
            # Get the max probability from symptom analysis
            max_symptom_prob = max(symptom_results['disease_probabilities'].values())
            
            # Fusion risk score (weighted average)
            fusion_risk = (sensor_risk * 0.6) + (max_symptom_prob * 0.4)
            
            # Determine primary disease
            primary_disease = symptom_results['primary_disease']
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sensor Risk Score", f"{sensor_risk:.3f}")
                
            with col2:
                st.metric("Symptom Risk Score", f"{max_symptom_prob:.3f}")
                
            with col3:
                st.metric("Fusion Risk Score", f"{fusion_risk:.3f}")
                if fusion_risk > 0.7:
                    st.error("🚨 *High risk detected!*")
                elif fusion_risk > 0.5:
                    st.warning("⚠ *Moderate risk detected*")
                else:
                    st.success("✅ *Low risk*")
            
            st.subheader("Primary Disease Prediction")
            st.info(f"*Based on fusion analysis, the primary concern is: {primary_disease.capitalize()}*")
            
            # Create new data entry
            new_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'location': latest_sensor_data.get('location', 'Unknown'),
                'pH': latest_sensor_data.get('pH', 0),
                'turbidity': latest_sensor_data.get('turbidity', 0),
                'TDS': latest_sensor_data.get('TDS', 0),
                'cl': latest_sensor_data.get('cl', 0),
                'dop': latest_sensor_data.get('dop', 0),
                'temp': latest_sensor_data.get('temp', 0),
                'risk_score': fusion_risk,
                'cholera': 1 if primary_disease == 'cholera' else 0,
                'dengue': 1 if primary_disease == 'dengue' else 0,
                'pneumonia': 1 if primary_disease == 'pneumonia' else 0,
                'typhoid': 1 if primary_disease == 'typhoid' else 0,
                'jaundice': 1 if primary_disease == 'jaundice' else 0,
                'probable_disease': primary_disease.capitalize(),
                'probability': max_symptom_prob,
                'num_water_sources': latest_sensor_data.get('num_water_sources', 1),
                'water_safety': 'Unsafe' if fusion_risk > 0.6 else 'Safe'
            }
            
            # Add to continuous learning data
            new_df = pd.DataFrame([new_entry])
            if st.session_state.continuous_learning_data.empty:
                st.session_state.continuous_learning_data = new_df
            else:
                st.session_state.continuous_learning_data = pd.concat(
                    [st.session_state.continuous_learning_data, new_df], 
                    ignore_index=True
                )
            
            # ✅ Save to DB
            try:
                conn = sqlite3.connect("water_data.db")
                cursor = conn.cursor()
                
                # Check if a similar entry already exists
                cursor.execute("""
                    SELECT COUNT(*) FROM sensor_data 
                    WHERE timestamp = ? AND location = ?
                """, (new_entry['timestamp'], new_entry['location']))
                
                count = cursor.fetchone()[0]
                
                if count == 0:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO sensor_data 
                        (timestamp, pH, turbidity, TDS, cl, dop, temp, risk_score, water_safety, 
                         cholera, dengue, pneumonia, typhoid, jaundice, probable_disease, 
                         probability, location, num_water_sources)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        new_entry['timestamp'],
                        new_entry['pH'],
                        new_entry['turbidity'],
                        new_entry['TDS'],
                        new_entry['cl'],
                        new_entry['dop'],
                        new_entry['temp'],
                        new_entry['risk_score'],
                        new_entry['water_safety'],
                        new_entry['cholera'],
                        new_entry['dengue'],
                        new_entry['pneumonia'],
                        new_entry['typhoid'],
                        new_entry['jaundice'],
                        new_entry['probable_disease'],
                        new_entry['probability'],
                        new_entry['location'],
                        new_entry['num_water_sources']
                    ))
                    
                    conn.commit()
                    st.success("Fusion data saved to database!")
                else:
                    st.warning("Similar entry already exists in the database.")
                
                conn.close()
            except Exception as e:
                st.error(f"Error saving fusion data: {e}")
            
            st.success("Fusion analysis complete! Data added to continuous learning dataset.")

# Dashboard page
elif page == "Dashboard":
    st.header("Analytics Dashboard")
    
    # ✅ Load data from database with proper timestamp handling
    try:
        conn = sqlite3.connect("water_data.db")
        db_df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
        conn.close()
        
        # Fix timestamp parsing
        if not db_df.empty and 'timestamp' in db_df.columns:
            try:
                db_df['timestamp'] = pd.to_datetime(db_df['timestamp'], utc=True)
            except:
                # If ISO format fails, try parsing without timezone
                db_df['timestamp'] = pd.to_datetime(db_df['timestamp'], errors='coerce')
        
        if not db_df.empty:
            st.subheader("Current Database Records")
            st.dataframe(db_df)
            
            # Check if we have data to display
            df = db_df
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Disease Distribution", "Risk Trends", "Location Analysis", "Sensor Correlations"])
            
            with tab1:
                st.subheader("Disease Distribution")
                
                if 'probable_disease' in df.columns:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    disease_counts = df['probable_disease'].value_counts()
                    wedges, texts, autotexts = ax.pie(disease_counts.values, labels=disease_counts.index, autopct='%1.1f%%')
                    
                    # Make the percentages larger
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(14)
                        autotext.set_weight('bold')
                    
                    # Make the labels larger
                    for text in texts:
                        text.set_fontsize(14)
                    
                    ax.set_title("Distribution of Predicted Diseases", fontsize=18, pad=20)
                    st.pyplot(fig)
                else:
                    st.warning("No disease data available for visualization.")
            
            with tab2:
                st.subheader("Risk Trends Over Time")
                
                if 'timestamp' in df.columns and 'risk_score' in df.columns:
                    # Ensure timestamp is in datetime format
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    
                    # Drop rows with invalid timestamps
                    df = df.dropna(subset=['timestamp'])
                    
                    time_series = df.set_index('timestamp')['risk_score']
                    
                    fig, ax = plt.subplots(figsize=(14, 8))
                    time_series.plot(ax=ax, linewidth=2.5)
                    ax.axhline(y=0.7, color='r', linestyle='--', label='High Risk Threshold', linewidth=2)
                    ax.axhline(y=0.5, color='y', linestyle='--', label='Moderate Risk Threshold', linewidth=2)
                    ax.set_ylabel("Risk Score", fontsize=16)
                    ax.set_xlabel("Time", fontsize=16)
                    ax.set_title("Risk Score Trend Over Time", fontsize=18, pad=20)
                    ax.legend(fontsize=14)
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    st.pyplot(fig)
                else:
                    st.warning("No timestamp or risk score data available for visualization.")
            
            with tab3:
                st.subheader("Analysis by Location")
                
                if 'location' in df.columns:
                    location_stats = df.groupby('location').agg({
                        'risk_score': 'mean',
                        'water_safety': lambda x: (x == 'Unsafe').mean() * 100
                    }).reset_index()
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
                    
                    # Average risk by location
                    location_stats_sorted = location_stats.sort_values('risk_score', ascending=False)
                    bars1 = ax1.bar(location_stats_sorted['location'], location_stats_sorted['risk_score'])
                    ax1.set_xticklabels(location_stats_sorted['location'], rotation=45, ha='right', fontsize=12)
                    ax1.set_ylabel("Average Risk Score", fontsize=16)
                    ax1.set_title("Average Risk Score by Location", fontsize=18, pad=20)
                    
                    # Add value labels on top of bars
                    for bar in bars1:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                    
                    # Unsafe percentage by location
                    location_stats_sorted = location_stats.sort_values('water_safety', ascending=False)
                    bars2 = ax2.bar(location_stats_sorted['location'], location_stats_sorted['water_safety'])
                    ax2.set_xticklabels(location_stats_sorted['location'], rotation=45, ha='right', fontsize=12)
                    ax2.set_ylabel("% Unsafe Water Sources", fontsize=16)
                    ax2.set_title("Unsafe Water Percentage by Location", fontsize=18, pad=20)
                    
                    # Add value labels on top of bars
                    for bar in bars2:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Alert for high-risk locations with larger font
                    high_risk_locations = location_stats[location_stats['risk_score'] > 0.7]
                    if not high_risk_locations.empty:
                        st.markdown("<h3 style='color: red; font-size: 20px;'>🚨 High risk detected in the following locations:</h3>", unsafe_allow_html=True)
                        for _, row in high_risk_locations.iterrows():
                            st.markdown(f"<p style='font-size: 18px;'>- {row['location']}: Risk score = <b>{row['risk_score']:.3f}</b></p>", unsafe_allow_html=True)
                else:
                    st.warning("No location data available for visualization.")
            
            with tab4:
                st.subheader("Sensor Data Correlations")
                
                sensor_features = ['pH', 'turbidity', 'TDS', 'cl', 'dop', 'temp']
                available_sensors = [f for f in sensor_features if f in df.columns]
                
                if available_sensors and 'risk_score' in df.columns:
                    # Calculate correlations with risk score
                    correlations = {}
                    for sensor in available_sensors:
                        correlation = df[sensor].corr(df['risk_score'])
                        correlations[sensor] = correlation
                    
                    correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
                    correlation_df = correlation_df.sort_values('Correlation', key=abs, ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = correlation_df['Correlation'].plot(kind='bar', ax=ax, color=['red' if x < 0 else 'green' for x in correlation_df['Correlation']])
                    ax.set_ylabel("Correlation with Risk Score", fontsize=16)
                    ax.set_title("Sensor Parameter Correlations with Risk", fontsize=18, pad=20)
                    plt.xticks(rotation=45, fontsize=14)
                    plt.yticks(fontsize=14)
                    
                    # Add value labels on top of bars
                    for i, bar in enumerate(bars.patches):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.05),
                                f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=12)
                    
                    st.pyplot(fig)
                    
                    st.write("Correlation details:")
                    st.dataframe(correlation_df.style.format({'Correlation': '{:.3f}'}))
                else:
                    st.warning("Insufficient data for correlation analysis.")
        else:
            st.warning("No data available for dashboard. Please perform some analyses first.")
    except Exception as e:
        st.error(f"Error reading from database: {e}")

# Data Management page
elif page == "Data Management":
    st.header("Data Management")
    
    st.info("Manage the data collected through analyses for continuous learning.")
    
    # ✅ Load data from database with proper timestamp handling
    try:
        conn = sqlite3.connect("water_data.db")
        db_df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
        conn.close()
        
        # Fix timestamp parsing
        if not db_df.empty and 'timestamp' in db_df.columns:
            try:
                db_df['timestamp'] = pd.to_datetime(db_df['timestamp'], utc=True)
            except:
                # If ISO format fails, try parsing without timezone
                db_df['timestamp'] = pd.to_datetime(db_df['timestamp'], errors='coerce')
        
        if not db_df.empty:
            st.subheader("Current Database Records")
            st.dataframe(db_df)
            
            # Display statistics with larger font
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Records", len(db_df))
                unsafe_count = (db_df['water_safety'] == 'Unsafe').sum() if 'water_safety' in db_df.columns else 0
                st.metric("Unsafe Water Sources", f"{unsafe_count} ({unsafe_count/len(db_df)*100:.1f}%)")
            
            with col2:
                avg_risk = db_df['risk_score'].mean() if 'risk_score' in db_df.columns else 0
                st.metric("Average Risk Score", f"{avg_risk:.3f}")
            
            # Download data with unique key
            if not db_df.empty:
                csv = db_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Full Database as CSV", 
                    csv, 
                    "sensor_database.csv", 
                    "text/csv",
                    key="download_full_database"  # Unique key added
                )
        else:
            st.info("Database is empty. Upload or add data first.")
    except Exception as e:
        st.error(f"Error reading from database: {e}")
    
    # Clear data - Fixed functionality
    if st.button("Clear All Data", key="clear_all_data"):
        try:
            conn = sqlite3.connect("water_data.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sensor_data")
            conn.commit()
            conn.close()
            st.success("Database cleared successfully!")
            # Use JavaScript to reload the page
            st.markdown("<meta http-equiv='refresh' content='0'>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error clearing database: {e}")

# Footer with larger font
st.sidebar.divider()
st.sidebar.markdown("""
<style>
.sidebar-box {
    background-color: #f1f5f9;
    border-radius: 12px;
    padding: 15px;
    font-size: 15px;
    font-family: 'Segoe UI', sans-serif;
    line-height: 1.5;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.05);
    margin-top: 15px;
}
.sidebar-box h4 {
    text-align: center;
    color: #0f172a;
    margin-bottom: 8px;
}
.sidebar-box ul {
    list-style-type: none;
    padding-left: 0;
    margin: 0;
}
.sidebar-box li {
    padding: 2px 0;
}
.team-name {
    font-weight: bold;
    color: #0284c7;
    # text-align: center;
    display: block;
    margin-top: 10px;
}
</style>

<div class="sidebar-box">
💬 <b>For questions or support:</b><br>
Please contact the development team.<br>
<i>This is a prototype for demonstration purposes.</i>

<span class="team-name">👨‍💻 Made by Team @ Code Catalysts</span>

<ul>
<li>• Debayan Das</li>
<li>• Akash Chakraborty</li>
<li>• Angana Das</li>
<li>• Aritri</li>
<li>• Agni</li>
<li>• Abhirup</li>
</ul>
</div>
""", unsafe_allow_html=True)
