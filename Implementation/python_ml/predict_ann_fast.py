import os
import sys
import json
import numpy as np
import time
import joblib
import tensorflow as tf

# Disable GPU for consistency
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Global model cache
_model_cache = {}

def get_cached_model():
    """Load and cache model components for reuse"""
    global _model_cache
    
    if not _model_cache:
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "ann_model.keras")
        scaler_path = os.path.join(script_dir, "ann_scaler.joblib")
        imputer_path = os.path.join(script_dir, "ann_imputer.joblib")
        config_path = os.path.join(script_dir, "ann_config.json")
        
        print("Loading model components into cache...")
        start = time.time()
        
        _model_cache['model'] = tf.keras.models.load_model(model_path)
        _model_cache['scaler'] = joblib.load(scaler_path)
        _model_cache['imputer'] = joblib.load(imputer_path)
        
        with open(config_path, 'r') as f:
            _model_cache['config'] = json.load(f)
        
        load_time = (time.time() - start) * 1000
        print(f"Model loaded and cached in {load_time:.2f}ms")
    
    return _model_cache

def extract_vibration_features_fast(vibration_series):
    """Optimized feature extraction"""
    # Convert to numpy array efficiently
    data = np.asarray(vibration_series, dtype=np.float32)
    
    # Handle NaN values efficiently
    mask = ~np.isnan(data)
    if not np.any(mask):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    clean_data = data[mask]
    if len(clean_data) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Vectorized calculations
    features = np.array([
        np.mean(clean_data),
        np.std(clean_data),
        np.max(clean_data),
        np.min(clean_data),
        len(clean_data) / len(data)
    ], dtype=np.float32)
    
    return features

def predict_fault_fast(input_file):
    """Optimized prediction with cached model"""
    start_time = time.time()
    
    try:
        # Get cached model components
        cache = get_cached_model()
        model = cache['model']
        scaler = cache['scaler'] 
        imputer = cache['imputer']
        threshold = cache['config'].get('threshold', 0.5)
        
        # Load input data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Extract data
        vibration = data.get('vibration', [0] * 100)
        temp = float(data.get('temp', 50.0))
        voltage = float(data.get('voltage', 220.0))
        
        # Fast feature extraction
        vib_features = extract_vibration_features_fast(vibration)
        
        # Create feature vector efficiently
        features = np.array([[temp, voltage] + vib_features.tolist()], dtype=np.float32)
        
        # Preprocessing
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        # Fast prediction
        prob = float(model.predict(features_scaled, verbose=0)[0][0])
        fault = 1 if prob > threshold else 0
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "fault": fault,
            "prob": prob,
            "latency_ms": latency,
            "method": "ann_cached",
            "threshold": threshold,
            "temp": temp,
            "voltage": voltage
        }
        
    except Exception as e:
        # Fallback
        latency = (time.time() - start_time) * 1000
        return {
            "fault": 0,
            "prob": 0.0,
            "latency_ms": latency,
            "method": "fallback",
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python predict_ann_fast.py <input_file>"}))
        sys.exit(1)
    
    input_file = sys.argv[1]
    result = predict_fault_fast(input_file)
    print(json.dumps(result))
