#!/usr/bin/env python3
"""
Quick test version of fog computing simulation
"""

from config_reader import ConfigReader
import time

def test_fog_simulation():
    print("🌐 Quick Fog Computing Test")
    print("=" * 50)
    
    try:
        # Test configuration loading
        print("1. Testing configuration...")
        config = ConfigReader()
        print("   ✓ Configuration loaded")
        
        # Display configuration
        print(f"   Machines: {config.get_num_machines()}")
        print(f"   Simulation Time: {config.get_sim_duration()}s")
        print(f"   Monitor Interval: {config.get_monitor_interval()}s")
        print(f"   Edge Threshold: {config.get_edge_threshold()}")
        print(f"   Python Exec: {config.get_python_executable()}")
        
        # Test model file existence
        print("\n2. Testing model files...")
        import os
        ann_exists = os.path.exists("python_ml/ann_model.keras")
        rf_exists = os.path.exists("python_ml/rf_model.pkl")
        print(f"   ANN model: {'✓' if ann_exists else '✗'}")
        print(f"   RF model: {'✓' if rf_exists else '✗'}")
        
        # Test a quick prediction
        print("\n3. Testing quick prediction...")
        import json
        import subprocess
        
        test_data = {
            "temp": 55.0,
            "voltage": 220.0,
            "vibration_x": 1.0,
            "vibration_y": 1.2,
            "vibration_z": 0.8,
            "current": 15.0,
            "true_fault": 1
        }
        
        with open("test_temp.json", "w") as f:
            json.dump(test_data, f)
        
        try:
            result = subprocess.run([config.get_python_executable(), 'python_ml/predict_ann.py', 'test_temp.json'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                prediction = json.loads(result.stdout)
                print(f"   ✓ ANN prediction: {prediction['prediction']} (prob: {prediction['prob']:.3f})")
            else:
                print(f"   ✗ ANN prediction failed: {result.stderr}")
        except Exception as e:
            print(f"   ✗ ANN prediction error: {e}")
        
        # Cleanup
        if os.path.exists("test_temp.json"):
            os.remove("test_temp.json")
        
        print("\n🎯 Quick test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fog_simulation()
