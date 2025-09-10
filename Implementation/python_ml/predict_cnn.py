import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def predict_fault(input_file):
    start_time = time.time()
    
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "cnn_model.h5")
    
    if not os.path.exists(model_path):
        return {"fault": 0, "prob": 0.1, "latency_ms": 1.0, "method": "fallback"}
    
    try:
        model = load_model(model_path)
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        vibration = data.get('vibration', [0] * 100)
        temp = data.get('temp', 50.0)
        voltage = data.get('voltage', 220.0)
        
        vibration_array = np.array(vibration, dtype=np.float32)
        
        if np.any(np.isnan(vibration_array)):
            mean_val = np.nanmean(vibration_array)
            vibration_array = np.nan_to_num(vibration_array, nan=mean_val)
        
        X = vibration_array.reshape(1, 100, 1)
        
        prob = float(model.predict(X, verbose=0)[0][0])
        fault = 1 if prob > 0.5 else 0
        
        latency = (time.time() - start_time) * 1000  
        
        return {
            "fault": fault,
            "prob": prob,
            "latency_ms": latency,
            "method": "cnn",
            "temp": temp,
            "voltage": voltage
        }
        
    except Exception as e:
        temp = data.get('temp', 50.0)
        voltage = data.get('voltage', 220.0)
        
        fault = 1 if (temp > 65 or voltage > 250 or voltage < 200) else 0
        prob = 0.8 if fault == 1 else 0.2
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "fault": fault,
            "prob": prob,
            "latency_ms": latency,
            "method": "fallback_threshold",
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python predict_cnn.py <input_file>"}))
        sys.exit(1)
    
    input_file = sys.argv[1]
    result = predict_fault(input_file)
    print(json.dumps(result))
