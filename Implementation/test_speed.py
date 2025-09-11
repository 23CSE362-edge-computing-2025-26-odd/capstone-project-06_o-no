import time
import json
import subprocess
import os

def compare_prediction_speeds():
    """Compare original vs optimized prediction speeds"""
    
    # Create test data
    test_data = {
        "temp": 65.0,
        "voltage": 245.0,
        "vibration": [0.5, -0.3, 1.2, 0.8, -0.9] * 20  # 100 points
    }
    
    test_file = "speed_test.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    print("=== PREDICTION SPEED COMPARISON ===")
    
    # Test original version
    print("\n1. Original Version (predict_ann.py):")
    original_times = []
    
    for i in range(5):
        start = time.time()
        result = subprocess.run(['python', 'python_ml/predict_ann.py', test_file], 
                              capture_output=True, text=True)
        elapsed = (time.time() - start) * 1000
        original_times.append(elapsed)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            print(f"  Run {i+1}: {elapsed:.1f}ms (reported: {data.get('latency_ms', 0):.1f}ms)")
        else:
            print(f"  Run {i+1}: {elapsed:.1f}ms (ERROR)")
    
    avg_original = sum(original_times) / len(original_times)
    
    # Test fast version  
    print("\n2. Optimized Version (predict_ann_fast.py):")
    fast_times = []
    
    for i in range(5):
        start = time.time()
        result = subprocess.run(['python', 'python_ml/predict_ann_fast.py', test_file],
                              capture_output=True, text=True)
        elapsed = (time.time() - start) * 1000
        fast_times.append(elapsed)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            print(f"  Run {i+1}: {elapsed:.1f}ms (reported: {data.get('latency_ms', 0):.1f}ms)")
        else:
            print(f"  Run {i+1}: {elapsed:.1f}ms (ERROR)")
    
    avg_fast = sum(fast_times) / len(fast_times)
    
    # Results
    print(f"\n=== RESULTS ===")
    print(f"Original average: {avg_original:.1f}ms")
    print(f"Optimized average: {avg_fast:.1f}ms")
    print(f"Speed improvement: {avg_original/avg_fast:.1f}x faster")
    print(f"Latency reduction: {avg_original - avg_fast:.1f}ms")
    
    # Cleanup
    os.remove(test_file)

if __name__ == "__main__":
    compare_prediction_speeds()
