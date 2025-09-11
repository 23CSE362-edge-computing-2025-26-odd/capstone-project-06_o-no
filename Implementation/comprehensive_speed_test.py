import time
import json
import subprocess
import os

def comprehensive_speed_test():
    """Compare all prediction approaches"""
    
    # Create test data
    test_data = {
        "temp": 65.0,
        "voltage": 245.0,
        "vibration": [0.5, -0.3, 1.2, 0.8, -0.9] * 20  # 100 points
    }
    
    test_file = "comprehensive_test.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    print("=== COMPREHENSIVE SPEED COMPARISON ===")
    
    approaches = [
        ("Original ANN (TensorFlow)", "python_ml/predict_ann.py"),
        ("Ultra-Fast Rules", "python_ml/predict_ultra_fast.py"),
    ]
    
    results = {}
    
    for name, script in approaches:
        print(f"\n{name}:")
        times = []
        latencies = []
        
        for i in range(3):  # 3 runs each
            start = time.time()
            result = subprocess.run(['python', script, test_file], 
                                  capture_output=True, text=True)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    reported_latency = data.get('latency_ms', 0)
                    latencies.append(reported_latency)
                    prob = data.get('prob', 0)
                    fault = data.get('fault', 0)
                    method = data.get('method', 'unknown')
                    print(f"  Run {i+1}: {elapsed:.1f}ms total, {reported_latency:.3f}ms internal")
                    print(f"    Result: fault={fault}, prob={prob:.4f}, method={method}")
                except:
                    print(f"  Run {i+1}: {elapsed:.1f}ms (parsing error)")
            else:
                print(f"  Run {i+1}: {elapsed:.1f}ms (execution error)")
        
        if times and latencies:
            results[name] = {
                'avg_total': sum(times) / len(times),
                'avg_internal': sum(latencies) / len(latencies),
                'min_total': min(times),
                'min_internal': min(latencies)
            }
    
    # Summary
    print(f"\n=== PERFORMANCE SUMMARY ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Average total time: {metrics['avg_total']:.1f}ms")
        print(f"  Average internal latency: {metrics['avg_internal']:.3f}ms")
        print(f"  Best total time: {metrics['min_total']:.1f}ms")
        print(f"  Best internal latency: {metrics['min_internal']:.3f}ms")
    
    # Speed comparison
    if len(results) >= 2:
        baseline = list(results.values())[0]['avg_total']
        fastest = list(results.values())[1]['avg_total']
        speedup = baseline / fastest
        print(f"\nSpeedup: {speedup:.1f}x faster ({baseline:.1f}ms → {fastest:.1f}ms)")
        latency_reduction = baseline - fastest
        print(f"Latency reduction: {latency_reduction:.1f}ms")
    
    # Cleanup
    os.remove(test_file)

if __name__ == "__main__":
    comprehensive_speed_test()
