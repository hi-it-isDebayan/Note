# generate data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Village coordinates for Northeast India
VILLAGE_COORDINATES = {
    "Mawlynnong": (25.2017637, 91.9133333),
    "Sohra": (25.2840000, 91.7210000),
    "Khonoma": (25.6523425, 94.0229921),
    "Ziro": (27.5949700, 93.8385400),
    "Majuli": (26.9500000, 94.1666670),
    "Haflong": (25.1690000, 93.0160000),
    "Mon": (26.7500000, 95.1000000),
    "Ukhrul": (25.3000000, 94.4500000),
    "Tawang": (27.5883300, 91.8652800),
    "Dawki": (25.2071951, 92.0095675),
    "Zokhawthar": (23.3659500, 93.3838980),
    "Pelling": (27.3015800, 88.2326600),
    "Aizawl": (23.7271060, 92.7176360),
    "Old Ziro": (27.5288110, 93.9094300),
}

WATER_SOURCE_TYPES = ["Tap", "Well", "Spring", "River", "Pond"]
WATER_TREATMENT_METHODS = ["None", "Boiling", "Chlorination", "Filter", "Other"]

def generate_water_data(n_villages=10, n_days=30, n_water_sources=3, include_weather=True):
    """
    Generate synthetic water quality data for villages in Northeast India
    
    Parameters:
    n_villages: Number of villages to generate data for
    n_days: Number of days to generate data for
    n_water_sources: Number of water sources per village
    include_weather: Whether to include weather data
    """
    # Select random villages
    villages = list(VILLAGE_COORDINATES.keys())[:n_villages]
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    data = []
    
    for date in dates:
        for village in villages:
            # Base water quality parameters with some randomness
            base_pH = np.random.uniform(6.8, 7.5)
            base_turbidity = np.random.uniform(1.0, 5.0)
            base_TDS = np.random.uniform(150, 400)
            base_chlorine = np.random.uniform(0.2, 0.8)
            base_temperature = np.random.uniform(20, 30)
            
            # Generate multiple readings per day (morning, afternoon, evening)
            for reading_time in ["morning", "afternoon", "evening"]:
                # Add some variation based on time of day
                if reading_time == "morning":
                    pH = base_pH - np.random.uniform(0.1, 0.3)
                    turbidity = base_turbidity + np.random.uniform(0.5, 2.0)
                    temperature = base_temperature - np.random.uniform(2, 5)
                elif reading_time == "afternoon":
                    pH = base_pH
                    turbidity = base_turbidity
                    temperature = base_temperature + np.random.uniform(2, 5)
                else:  # evening
                    pH = base_pH + np.random.uniform(0.1, 0.3)
                    turbidity = base_turbidity - np.random.uniform(0.5, 1.5)
                    temperature = base_temperature - np.random.uniform(1, 3)
                
                # Add some random daily variation
                pH += np.random.uniform(-0.2, 0.2)
                turbidity += np.random.uniform(-0.5, 0.5)
                TDS = base_TDS + np.random.uniform(-20, 20)
                residual_chlorine = base_chlorine + np.random.uniform(-0.1, 0.1)
                temperature += np.random.uniform(-1, 1)
                
                # Ensure values are within reasonable bounds
                pH = max(6.0, min(8.5, pH))
                turbidity = max(0.1, min(50.0, turbidity))
                TDS = max(50, min(1000, TDS))
                residual_chlorine = max(0.1, min(2.0, residual_chlorine))
                temperature = max(15, min(35, temperature))
                
                # Generate rainfall data if requested
                rainfall = 0
                if include_weather:
                    # Higher probability of rainfall in Northeast India
                    if np.random.random() < 0.4:  # 40% chance of rain
                        rainfall = np.random.uniform(5, 50)  # mm
                
                # Calculate risk score
                risk_score = calculate_risk_score(pH, turbidity, TDS, residual_chlorine)
                
                # Determine water safety
                water_safety = "safe" if risk_score < 30 else "unsafe"
                
                # Select random water source and treatment
                water_source = np.random.choice(WATER_SOURCE_TYPES)
                water_treatment = np.random.choice(WATER_TREATMENT_METHODS)
                
                # Generate sanitation data
                sanitation_facilities = np.random.randint(30, 90)  # percentage
                open_defecation = np.random.randint(5, 40)  # percentage
                waste_management = np.random.randint(3, 9)  # scale of 1-10
                
                # Create record
                record = {
                    "timestamp": date + timedelta(hours=np.random.randint(6, 20)),
                    "location": village,
                    "pH": round(pH, 2),
                    "turbidity": round(turbidity, 2),
                    "TDS": round(TDS, 1),
                    "residual_chlorine": round(residual_chlorine, 2),
                    "temperature": round(temperature, 1),
                    "rainfall": round(rainfall, 1),
                    "water_source_type": water_source,
                    "water_treatment": water_treatment,
                    "sanitation_facilities": sanitation_facilities,
                    "open_defecation": open_defecation,
                    "waste_management": waste_management,
                    "risk_score": round(risk_score, 2),
                    "water_safety": water_safety
                }
                
                data.append(record)
    
    return pd.DataFrame(data)

def calculate_risk_score(pH, turbidity, TDS, residual_chlorine):
    """
    Calculate a water quality risk score based on multiple parameters
    
    Parameters:
    pH: pH value
    turbidity: Turbidity in NTU
    TDS: Total Dissolved Solids in ppm
    residual_chlorine: Residual chlorine in mg/L
    
    Returns:
    risk_score: A score from 0-100 indicating water quality risk
    """
    # pH component (ideal is 7)
    pH_score = min(100, abs(pH - 7) / 0.5 * 25)
    
    # Turbidity component (lower is better)
    turbidity_score = min(100, turbidity / 5 * 25)
    
    # TDS component (higher is worse)
    TDS_score = min(100, TDS / 1000 * 25)
    
    # Chlorine component (both too low and too high are bad)
    if residual_chlorine < 0.2:
        chlorine_score = (0.2 - residual_chlorine) / 0.2 * 25
    elif residual_chlorine > 2.0:
        chlorine_score = (residual_chlorine - 2.0) / 2.0 * 25
    else:
        chlorine_score = 0
    
    # Total risk score
    risk_score = pH_score + turbidity_score + TDS_score + chlorine_score
    
    return min(100, risk_score)

# For testing
if __name__ == "__main__":
    df = generate_water_data()
    print(f"Generated {len(df)} records")
    print(df.head())
    df.to_csv("indian_tribal_water_metrics_2025.csv", index=False)