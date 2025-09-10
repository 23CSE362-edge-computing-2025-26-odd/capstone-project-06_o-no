import os
import sys
import json
import numpy as np
import pickle
import time

def predict_fault(input_file):
    start_time = time.time()
    
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "rf_model.pkl")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return {"fault": 0, "prob": 0.1, "latency_ms": 1.0, "method": "file_error"}
    
    if not os.path.exists(model_path):
        return {"fault": 0, "prob": 0.1, "latency_ms": 1.0, "method": "fallback"}
    
    try:
        import joblib
        model = joblib.load(model_path)
        
        vibration = data.get('vibration', [0] * 100)
        temp = data.get('temp', 50.0)
        voltage = data.get('voltage', 220.0)
        
        vibration_array = np.array(vibration, dtype=np.float32)
        
        if np.any(np.isnan(vibration_array)):
            mean_val = np.nanmean(vibration_array)
            vibration_array = np.nan_to_num(vibration_array, nan=mean_val)
        
        vib_mean = np.mean(vibration_array)
        vib_std = np.std(vibration_array)
        vib_max = np.max(vibration_array)
        vib_min = np.min(vibration_array)
        vib_rms = np.sqrt(np.mean(vibration_array**2))
        
        features = np.array([[vib_mean, temp, voltage]])
        
        if np.isnan(temp):
            features[0][1] = 50.0
        if np.isnan(voltage):
            features[0][2] = 220.0
            
        prob = float(model.predict_proba(features)[0][1])  
        fault = 1 if prob > 0.5 else 0
        
        latency = (time.time() - start_time) * 1000  
        
        return {
            "fault": fault,
            "prob": prob,
            "latency_ms": float(latency),
            "method": "random_forest",
            "temp": float(temp),
            "voltage": float(voltage),
            "features": {
                "vib_mean": float(vib_mean),
                "vib_std": float(vib_std),
                "vib_rms": float(vib_rms)
            }
        }
        
    except Exception as e:
        temp = data.get('temp', 50.0)
        voltage = data.get('voltage', 220.0)
        vibration = data.get('vibration', [0] * 100)
        
        vibration_array = np.array(vibration, dtype=np.float32)
        if np.any(np.isnan(vibration_array)):
            mean_val = np.nanmean(vibration_array)
            vibration_array = np.nan_to_num(vibration_array, nan=mean_val)
        
        vib_rms = np.sqrt(np.mean(vibration_array**2))
        
        fault = 1 if (temp > 70 or voltage > 260 or voltage < 180 or vib_rms > 3.0) else 0
        prob = 0.9 if fault == 1 else 0.1
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "fault": fault,
            "prob": prob,
            "latency_ms": float(latency),
            "method": "fallback_advanced_threshold",
            "error": str(e),
            "features": {
                "vib_rms": float(vib_rms),
                "temp": float(temp),
                "voltage": float(voltage)
            }
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python predict_rf.py <input_file>"}))
        sys.exit(1)
    
    input_file = sys.argv[1]
    result = predict_fault(input_file)
    print(json.dumps(result))
