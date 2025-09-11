import time
import json
import os
import sys
import numpy as np
import joblib
import tensorflow as tf

# Disable GPU for consistency
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def profile_prediction_latency():
    """Profile each component of the prediction pipeline"""
    
    # Create test data
    test_data = {
        "temp": 65.0,
        "voltage": 245.0,
        "vibration": [np.random.normal(0, 1.5) for _ in range(100)]
    }
    
    # Write test file
    test_file = "test_latency.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    print("=== LATENCY PROFILING ===")
    
    total_start = time.time()
    
    # 1. File I/O
    start = time.time()
    with open(test_file, 'r') as f:
        data = json.load(f)
    file_io_time = (time.time() - start) * 1000
    print(f"1. File I/O: {file_io_time:.2f}ms")
    
    # 2. Feature extraction
    start = time.time()
    vibration = data.get('vibration', [0] * 100)
    temp = data.get('temp', 50.0)
    voltage = data.get('voltage', 220.0)
    
    # Extract vibration features
    if isinstance(vibration, list):
        vib_data = np.array(vibration, dtype=float)
    else:
        vib_data = np.array(vibration, dtype=float)
    
    clean_data = vib_data[~np.isnan(vib_data)]
    vib_features = [
        np.mean(clean_data),
        np.std(clean_data),
        np.max(clean_data),
        np.min(clean_data),
        len(clean_data) / len(vib_data)
    ]
    
    features = np.array([[temp, voltage] + vib_features], dtype=np.float32)
    feature_time = (time.time() - start) * 1000
    print(f"2. Feature extraction: {feature_time:.2f}ms")
    
    # 3. Model loading
    start = time.time()
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "python_ml", "ann_model.keras")
    scaler_path = os.path.join(script_dir, "python_ml", "ann_scaler.joblib")
    imputer_path = os.path.join(script_dir, "python_ml", "ann_imputer.joblib")
    config_path = os.path.join(script_dir, "python_ml", "ann_config.json")
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_loading_time = (time.time() - start) * 1000
    print(f"3. Model loading: {model_loading_time:.2f}ms")
    
    # 4. Preprocessing
    start = time.time()
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)
    preprocessing_time = (time.time() - start) * 1000
    print(f"4. Preprocessing: {preprocessing_time:.2f}ms")
    
    # 5. Model inference
    start = time.time()
    prob = float(model.predict(features_scaled, verbose=0)[0][0])
    inference_time = (time.time() - start) * 1000
    print(f"5. Model inference: {inference_time:.2f}ms")
    
    # 6. Post-processing
    start = time.time()
    threshold = config.get('threshold', 0.5)
    fault = 1 if prob > threshold else 0
    postprocessing_time = (time.time() - start) * 1000
    print(f"6. Post-processing: {postprocessing_time:.2f}ms")
    
    total_time = (time.time() - total_start) * 1000
    print(f"\nTOTAL TIME: {total_time:.2f}ms")
    
    # Now test with pre-loaded model (simulating cached models)
    print("\n=== OPTIMIZED LATENCY (Pre-loaded models) ===")
    
    # Test multiple predictions with pre-loaded models
    times = []
    for i in range(10):
        start = time.time()
        
        # Only feature extraction + inference (no model loading)
        vib_data = np.array(test_data['vibration'], dtype=float)
        clean_data = vib_data[~np.isnan(vib_data)]
        vib_features = [
            np.mean(clean_data),
            np.std(clean_data), 
            np.max(clean_data),
            np.min(clean_data),
            len(clean_data) / len(vib_data)
        ]
        
        features = np.array([[test_data['temp'], test_data['voltage']] + vib_features], dtype=np.float32)
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        prob = float(model.predict(features_scaled, verbose=0)[0][0])
        fault = 1 if prob > threshold else 0
        
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    avg_optimized_time = np.mean(times)
    print(f"Average optimized latency: {avg_optimized_time:.2f}ms")
    print(f"Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")
    
    # Cleanup
    os.remove(test_file)
    
    print(f"\n=== ANALYSIS ===")
    print(f"Current simulation latency: ~350ms")
    print(f"Profiled latency: {total_time:.2f}ms")
    print(f"Optimized latency: {avg_optimized_time:.2f}ms")
    print(f"Model loading overhead: {model_loading_time:.2f}ms ({model_loading_time/total_time*100:.1f}%)")

if __name__ == "__main__":
    profile_prediction_latency()
