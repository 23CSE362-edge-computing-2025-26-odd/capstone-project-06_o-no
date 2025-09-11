import json
import sys
import time
import numpy as np

def extract_features_fast(vibration, temp, voltage):
    """Lightning fast feature extraction"""
    # Simple statistical features
    vib_array = np.array(vibration, dtype=np.float32)
    
    # Basic stats
    vib_mean = np.mean(vib_array)
    vib_std = np.std(vib_array) 
    vib_max = np.max(vib_array)
    vib_energy = np.sum(vib_array ** 2)
    
    return [temp, voltage, vib_mean, vib_std, vib_max, vib_energy]

def predict_fault_ultra_fast(input_file):
    """Ultra-fast rule-based prediction (sub-millisecond)"""
    start_time = time.time()
    
    try:
        # Load data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        temp = data.get('temp', 50.0)
        voltage = data.get('voltage', 220.0)
        vibration = data.get('vibration', [0] * 100)
        
        # Extract features
        features = extract_features_fast(vibration, temp, voltage)
        temp, voltage, vib_mean, vib_std, vib_max, vib_energy = features
        
        # Simple but effective rule-based classification
        # Based on patterns learned from training data analysis
        fault_score = 0.0
        
        # Temperature factor (strong indicator)
        if temp > 60:
            fault_score += 0.4
        elif temp > 55:
            fault_score += 0.2
            
        # Voltage factor (strong indicator)
        if voltage > 235:
            fault_score += 0.4
        elif voltage < 210:
            fault_score += 0.3
            
        # Vibration factors
        if abs(vib_mean) > 1.0:
            fault_score += 0.1
        if vib_std > 2.0:
            fault_score += 0.1
        if abs(vib_max) > 3.0:
            fault_score += 0.1
        if vib_energy > 150:
            fault_score += 0.1
            
        # Convert to probability
        probability = float(min(fault_score, 1.0))
        fault = 1 if probability > 0.5 else 0
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "fault": fault,
            "prob": probability,
            "latency_ms": latency,
            "method": "ultra_fast_rules",
            "temp": float(temp),
            "voltage": float(voltage),
            "features": {
                "vib_mean": float(vib_mean),
                "vib_std": float(vib_std),
                "vib_max": float(vib_max),
                "vib_energy": float(vib_energy)
            }
        }
        
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return {
            "fault": 0,
            "prob": 0.0,
            "latency_ms": latency,
            "method": "ultra_fast_error", 
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python predict_ultra_fast.py <input_file>"}))
        sys.exit(1)
    
    input_file = sys.argv[1]
    result = predict_fault_ultra_fast(input_file)
    print(json.dumps(result))
