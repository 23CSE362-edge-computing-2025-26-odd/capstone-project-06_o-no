#!/usr/bin/env python3
"""
Test the ANN model with different input patterns to understand its behavior
"""

import json
import numpy as np
import pandas as pd
import ast
import joblib
import tensorflow as tf
from predict_ann import extract_vibration_features

# Load model and preprocessing objects
model = tf.keras.models.load_model("ann_model.keras")
scaler = joblib.load("ann_scaler.joblib")
imputer = joblib.load("ann_imputer.joblib")

def test_prediction(temp, voltage, vibration_pattern="normal"):
    """Test prediction with specific parameters"""
    if vibration_pattern == "normal":
        vibration = np.random.normal(0, 1.0, 100).tolist()
    else:  # fault
        vibration = np.random.normal(0, 2.5, 100).tolist()
        # Add some spikes
        spike_indices = np.random.choice(100, 10, replace=False)
        for idx in spike_indices:
            vibration[idx] += np.random.uniform(3, 6)
    
    # Extract features
    vib_features = extract_vibration_features(vibration)
    features = np.array([[temp, voltage] + vib_features], dtype=np.float32)
    
    # Preprocessing
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)
    
    # Prediction
    prob = float(model.predict(features_scaled, verbose=0)[0][0])
    fault = 1 if prob > 0.5 else 0
    
    return prob, fault

print("Testing ANN model behavior:")
print("=" * 50)

# Test with training data patterns
print("1. Training data patterns:")
print(f"Normal (temp=50, voltage=220): prob={test_prediction(50, 220, 'normal')[0]:.4f}")
print(f"Fault (temp=70, voltage=245):  prob={test_prediction(70, 245, 'fault')[0]:.4f}")

print("\n2. Edge cases:")
print(f"Borderline high temp (temp=60, voltage=220): prob={test_prediction(60, 220, 'normal')[0]:.4f}")
print(f"Borderline high voltage (temp=50, voltage=235): prob={test_prediction(50, 235, 'normal')[0]:.4f}")

print("\n3. Extreme cases:")
print(f"Very high temp (temp=80, voltage=220): prob={test_prediction(80, 220, 'normal')[0]:.4f}")
print(f"Very high voltage (temp=50, voltage=260): prob={test_prediction(50, 260, 'normal')[0]:.4f}")

print("\n4. Analysis of training data:")
df = pd.read_csv('synthetic_data.csv')
normal_data = df[df['label'] == 0]
fault_data = df[df['label'] == 1]

print(f"Normal samples: {len(normal_data)}")
print(f"  - Temp range: {normal_data['temp'].min():.1f} to {normal_data['temp'].max():.1f}")
print(f"  - Voltage range: {normal_data['voltage'].min():.1f} to {normal_data['voltage'].max():.1f}")

print(f"Fault samples: {len(fault_data)}")
print(f"  - Temp range: {fault_data['temp'].min():.1f} to {fault_data['temp'].max():.1f}")
print(f"  - Voltage range: {fault_data['voltage'].min():.1f} to {fault_data['voltage'].max():.1f}")

print("\n5. Clear separation thresholds:")
temp_threshold = (normal_data['temp'].max() + fault_data['temp'].min()) / 2
voltage_threshold = (normal_data['voltage'].max() + fault_data['voltage'].min()) / 2
print(f"Temp threshold: ~{temp_threshold:.1f}°C")
print(f"Voltage threshold: ~{voltage_threshold:.1f}V")
