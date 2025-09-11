import time
import json
import subprocess
import os

def comprehensive_latency_analysis():
    """Compare all available prediction methods"""
    
    # Create test data 
    test_data = {
        "temp": 65.0,
        "voltage": 245.0,
        "vibration": [0.5, -0.3, 1.2, 0.8, -0.9] * 20  # 100 points
    }
    
    test_file = "latency_test.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    print("=== COMPLETE LATENCY ANALYSIS ===")
    print("Testing all available prediction methods...\n")
    
    methods = [
        ("Original CNN", "python_ml/predict_cnn.py"),
        ("Random Forest", "python_ml/predict_rf.py"), 
        ("New ANN", "python_ml/predict_ann.py"),
        ("Ultra-Fast Rules", "python_ml/predict_ultra_fast.py"),
    ]
    
    results = {}
    
    for name, script in methods:
        print(f"🔄 Testing {name}...")
        times = []
        predictions = []
        
        # Test 3 runs to get average
        for i in range(3):
            start = time.time()
            result = subprocess.run(['python', script, test_file], 
                                  capture_output=True, text=True)
            total_elapsed = (time.time() - start) * 1000
            times.append(total_elapsed)
            
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    predictions.append(data)
                    internal_latency = data.get('latency_ms', 0)
                    fault = data.get('fault', 0)
                    prob = data.get('prob', 0)
                    method = data.get('method', 'unknown')
                    
                    print(f"  Run {i+1}: {total_elapsed:.1f}ms total, {internal_latency:.1f}ms internal")
                    print(f"    → fault={fault}, prob={prob:.4f}, method={method}")
                    
                except Exception as e:
                    print(f"  Run {i+1}: {total_elapsed:.1f}ms (parse error: {e})")
                    predictions.append(None)
            else:
                print(f"  Run {i+1}: {total_elapsed:.1f}ms (execution error)")
                print(f"    Error: {result.stderr}")
                predictions.append(None)
        
        # Calculate statistics
        valid_times = [t for t in times if t is not None]
        valid_predictions = [p for p in predictions if p is not None]
        
        if valid_times and valid_predictions:
            avg_total = sum(valid_times) / len(valid_times)
            min_total = min(valid_times)
            avg_internal = sum(p.get('latency_ms', 0) for p in valid_predictions) / len(valid_predictions)
            
            results[name] = {
                'avg_total': avg_total,
                'min_total': min_total,
                'avg_internal': avg_internal,
                'predictions': valid_predictions,
                'success_rate': len(valid_predictions) / 3 * 100
            }
        
        print()
    
    # Performance summary
    print("=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  ⏱️  Average total time: {metrics['avg_total']:.1f}ms")
        print(f"  ⚡ Best total time: {metrics['min_total']:.1f}ms") 
        print(f"  🔧 Average internal latency: {metrics['avg_internal']:.1f}ms")
        print(f"  ✅ Success rate: {metrics['success_rate']:.0f}%")
        
        # Show prediction consistency
        preds = metrics['predictions']
        if len(preds) > 0:
            faults = [p.get('fault', 0) for p in preds]
            probs = [p.get('prob', 0) for p in preds]
            methods = [p.get('method', 'unknown') for p in preds]
            print(f"  🎯 Predictions: fault={faults}, prob=[{min(probs):.3f}-{max(probs):.3f}]")
            print(f"  🏷️  Method: {methods[0]}")
    
    # Speed comparison
    print("\n" + "=" * 60)
    print("🏃 SPEED COMPARISON")
    print("=" * 60)
    
    if len(results) >= 2:
        # Sort by speed (fastest first)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_total'])
        
        fastest_name, fastest_metrics = sorted_results[0]
        print(f"\n🥇 Fastest: {fastest_name} ({fastest_metrics['avg_total']:.1f}ms)")
        
        for i, (name, metrics) in enumerate(sorted_results[1:], 1):
            speedup = metrics['avg_total'] / fastest_metrics['avg_total']
            slowdown = metrics['avg_total'] - fastest_metrics['avg_total']
            print(f"🥈 #{i+1}: {name} ({metrics['avg_total']:.1f}ms) - {speedup:.1f}x slower (+{slowdown:.0f}ms)")
    
    # Accuracy comparison  
    print("\n" + "=" * 60)
    print("🎯 PREDICTION COMPARISON") 
    print("=" * 60)
    print(f"Test case: temp={test_data['temp']}°C, voltage={test_data['voltage']}V")
    print("Expected: High temp + high voltage → likely FAULT")
    
    for name, metrics in results.items():
        if metrics['predictions']:
            pred = metrics['predictions'][0]  # Use first prediction
            fault = pred.get('fault', 0)
            prob = pred.get('prob', 0)
            status = "FAULT" if fault else "NORMAL"
            confidence = "High" if (prob > 0.7 or prob < 0.3) else "Medium" if (prob > 0.6 or prob < 0.4) else "Low"
            
            print(f"  {name}: {status} (prob={prob:.3f}, confidence={confidence})")
    
    # Cleanup
    os.remove(test_file)
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    print("For Edge Computing:")
    print("  🚀 Ultra-Fast Rules: Best for real-time edge with <1ms latency")
    print("  🔬 Random Forest: Good balance of speed and ML accuracy")
    print("  🧠 CNN/ANN: Best accuracy but higher latency")
    print("\nFor Cloud Computing:")
    print("  🧠 CNN/ANN: Full ML accuracy acceptable with cloud resources")
    print("  📊 Random Forest: Ensemble reliability with reasonable speed")

if __name__ == "__main__":
    comprehensive_latency_analysis()
