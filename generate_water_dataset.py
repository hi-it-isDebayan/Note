import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_water_data():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate tribal locations
    #3 character X, Y, Z and 3 * 20 = 60 * 365 days = 21900 rows       (say, 60 tribes, for now)
    tribes = [f"Tribe{letter}{i:02d}" for letter in ['X', 'Y', 'Z'] for i in range(1, 21)]


    # Create date range for 2025
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')

    # Generate synthetic data
    data = []
    for date in dates:
        for tribe in tribes:
            # Base parameters with some randomness
            pH = round(np.random.normal(7.2, 0.8), 1)
            turbidity = round(np.random.gamma(2, 2), 2)
            TDS = int(np.random.normal(500, 300))
            cl = round(np.random.uniform(0.01, 0.25), 2)
            dop = round(np.random.normal(6.0, 1.2), 1)
            temp = round(np.random.normal(26.0, 3.0), 1)
            
            # Calculate risk score based on parameters
            risk_score = int(
                10 * abs(pH - 7.0) + 
                0.5 * turbidity + 
                0.05 * max(0, TDS - 500) + 
                50 * max(0, 0.1 - cl) + 
                5 * max(0, 5.0 - dop) + 
                abs(temp - 25)
            )
            risk_score = min(100, risk_score)
            
            # Disease probabilities based on parameters
            cholera_prob = min(0.95, 0.6 * (turbidity > 15) + 0.4 * (cl < 0.05) + 0.3 * (pH > 8.2))
            dengue_prob = min(0.95, 0.7 * (turbidity > 10) + 0.5 * (temp > 28))
            typhoid_prob = min(0.95, 0.5 * (turbidity > 12) + 0.6 * (cl < 0.08))
            jaundice_prob = min(0.95, 0.4 * (turbidity > 8) + 0.5 * (TDS > 800))
            
            # Determine the most probable disease
            probs = [cholera_prob, dengue_prob, typhoid_prob, jaundice_prob]
            diseases = ['cholera', 'dengue', 'typhoid', 'jaundice']
            max_prob = max(probs)
            probable_disease = diseases[probs.index(max_prob)] if max_prob > 0.5 else 'low_risk'
            
            # Binary disease indicators (1 if probability > 0.6)
            cholera = 1 if cholera_prob > 0.6 else 0
            dengue = 1 if dengue_prob > 0.6 else 0
            typhoid = 1 if typhoid_prob > 0.6 else 0
            jaundice = 1 if jaundice_prob > 0.6 else 0
            
            # Pneumonia (not primarily water-borne, so lower probability)
            pneumonia = 1 if np.random.random() < 0.05 else 0
            
            # Number of water sources (1-6)
            num_water_sources = np.random.randint(1, 7)
            
            # Water safety (unsafe if risk_score > 50)
            water_safety = 'unsafe' if risk_score > 50 else 'safe'
            
            data.append([
                date.strftime('%Y-%m-%d'),
                pH, turbidity, TDS, cl, dop, temp,
                risk_score,
                cholera, dengue, pneumonia, typhoid, jaundice,
                probable_disease, max_prob if max_prob > 0.5 else max(probs),
                tribe,
                num_water_sources,
                water_safety
            ])

    # Create DataFrame
    columns = [
        'timestamp', 'pH', 'turbidity', 'TDS', 'cl', 'dop', 'temp',
        'risk_score', 'cholera', 'dengue', 'pneumonia', 'typhoid', 'jaundice',
        'probable_disease', 'probability', 'location', 'num_water_sources', 'water_safety'
    ]

    df = pd.DataFrame(data, columns=columns)
    return df

# This part will only run if the script is executed directly
if __name__ == "__main__":  # Fixed: Changed _name to _name_
    df = generate_water_data()
    
    # Define the folder and filename
    output_folder = 'Generate_data_set'
    output_filename = 'indian_tribal_water_metrics_2025.csv'

    # Create the folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create the full file path by joining the folder and filename
    output_path = os.path.join(output_folder, output_filename)

    # Save to CSV using the full path
    df.to_csv(output_path, index=False)

    print(f"Dataset created with {len(df)} rows and saved to '{output_path}'")