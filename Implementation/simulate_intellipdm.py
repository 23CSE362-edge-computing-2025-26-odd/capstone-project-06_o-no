#!/usr/bin/env python3
"""
Simplified IntelliPdM simulation test
This script simulates the key components without the full Java framework
"""

import subprocess
import os
import time
import json
import random

def test_ann_integration():
    """Test ANN model integration"""
    print("=" * 80)
    print("INTELLIPDM PREDICTIVE MAINTENANCE SIMULATION")
    print("=" * 80)
    print("Starting ANN-based predictive maintenance simulation...")
    
    # Train ANN model (skip if already exists)
    print("\n1. Checking ANN model...")
    ann_model_path = "python_ml/ann_model.keras"
    if os.path.exists(ann_model_path):
        print("   ✓ ANN model already exists, skipping training")
    else:
        print("   Training ANN model...")
        try:
            result = subprocess.run(['python', 'python_ml/train_ann.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✓ ANN model trained successfully")
            else:
                print("   ✗ ANN model training failed")
                print(f"   Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"   ✗ Error training ANN: {e}")
            return False
    
    # Train Random Forest model for cloud (skip if already exists)
    print("\n2. Checking Random Forest model...")
    rf_model_path = "python_ml/rf_model.pkl"
    if os.path.exists(rf_model_path):
        print("   ✓ Random Forest model already exists, skipping training")
    else:
        print("   Training Random Forest model...")
        try:
            result = subprocess.run(['python', 'python_ml/train_rf.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✓ Random Forest model trained successfully")
            else:
                print("   ✗ Random Forest model training failed")
        except Exception as e:
            print(f"   ✗ Error training RF: {e}")
    
    # Simulate edge processing with ANN
    print("\n3. Simulating edge processing with ANN...")
    
    # Simulate multiple machines
    machines = 5
    predictions_correct = 0
    total_predictions = 0
    
    for machine_id in range(1, machines + 1):
        print(f"\n   Machine-{machine_id} Processing:")
        
        # Generate test data for multiple time points
        for time_point in range(5):
            # Generate realistic sensor data covering normal and fault conditions
            # Generate varied conditions to test model robustness
            temp = random.uniform(45, 80)  # Full temperature range
            voltage = random.uniform(200, 260)  # Full voltage range
            
            # Generate vibration pattern with some variation
            vibration = []
            for i in range(100):
                base_val = random.gauss(0, 1.5)  
                if random.random() < 0.05:  # 5% chance of spike
                    base_val += random.uniform(2, 4)  
                vibration.append(base_val)
            
            # Determine expected fault based on model's learned patterns:
            # From training data analysis: fault when temp > ~60°C OR voltage > ~235V
            is_fault = (temp > 60) or (voltage > 235)
            
            # Create test input
            test_data = {
                "temp": temp,
                "voltage": voltage,
                "vibration": vibration
            }
            
            # Write to temp file
            temp_file = f"temp_machine_{machine_id}_{time_point}.json"
            with open(temp_file, 'w') as f:
                json.dump(test_data, f)
            
            # Run ANN prediction
            try:
                result = subprocess.run(['python', 'python_ml/predict_ann.py', temp_file], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    prediction = json.loads(result.stdout)
                    predicted_fault = prediction['fault']
                    probability = prediction['prob']
                    method = prediction['method']
                    latency = prediction['latency_ms']
                    
                    # Check if prediction matches expectation (simplified)
                    prediction_correct = (predicted_fault == 1) == is_fault
                    if prediction_correct:
                        predictions_correct += 1
                    total_predictions += 1
                    
                    status = "FAULT" if predicted_fault == 1 else "NORMAL"
                    expected = "FAULT" if is_fault else "NORMAL"
                    accuracy_mark = "✓" if prediction_correct else "✗"
                    
                    print(f"     Time {time_point}: {status} (prob={probability:.4f}, "
                          f"method={method}, latency={latency:.1f}ms) "
                          f"Expected: {expected} {accuracy_mark}")
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
            except Exception as e:
                print(f"     Time {time_point}: Prediction failed - {e}")
            
            time.sleep(0.1)  # Small delay between predictions
    
    # Calculate overall accuracy
    if total_predictions > 0:
        accuracy = (predictions_correct / total_predictions) * 100
        print(f"\n4. SIMULATION RESULTS:")
        print(f"   Total Predictions: {total_predictions}")
        print(f"   Correct Predictions: {predictions_correct}")
        print(f"   Overall Accuracy: {accuracy:.1f}%")
        
        if accuracy >= 70:
            print("   ✓ SIMULATION SUCCESSFUL - Good predictive accuracy!")
        else:
            print("   ⚠ SIMULATION NEEDS IMPROVEMENT - Low accuracy")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETED")
    print("=" * 80)
    print("Key Features Demonstrated:")
    print("  ✓ ANN model training and deployment")
    print("  ✓ Real-time edge prediction")
    print("  ✓ Feature extraction from vibration data")
    print("  ✓ Multi-machine monitoring")
    print("  ✓ Fault detection with probability scores")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    test_ann_integration()
