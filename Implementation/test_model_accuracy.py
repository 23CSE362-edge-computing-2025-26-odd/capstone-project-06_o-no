#!/usr/bin/env python3
"""
Test the ANN model accuracy with known patterns from training data
"""

import json
import pandas as pd
import numpy as np
import subprocess
import os
import tempfile

def test_model_with_training_patterns():
    """Test the model using actual patterns from training data"""
    
    print("=== TESTING ANN MODEL WITH KNOWN PATTERNS ===")
    
    # Load the training data to get real examples
    df = pd.read_csv('python_ml/clean_synthetic_data.csv')
    
    # Get some actual normal and fault samples
    normal_samples = df[df['label'] == 0].sample(10, random_state=42)
    fault_samples = df[df['label'] == 1].sample(10, random_state=42)
    
    correct_predictions = 0
    total_predictions = 0
    
    print("\n--- Testing NORMAL samples ---")
    for idx, row in normal_samples.iterrows():
        # Extract vibration data (it's stored as a string representation of a list)
        import ast
        vibration = ast.literal_eval(row['vibration'])
        
        # Create test data
        test_data = {
            "temp": row['temp'],
            "voltage": row['voltage'], 
            "vibration": vibration
        }
        
        # Test prediction
        result = test_single_prediction(test_data, expected_label=0)
        if result:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"  Sample {idx}: temp={row['temp']:.1f}°C, voltage={row['voltage']:.1f}V, "
              f"prediction={'CORRECT' if result else 'WRONG'}")
    
    print("\n--- Testing FAULT samples ---")
    for idx, row in fault_samples.iterrows():
        # Extract vibration data (it's stored as a string representation of a list)
        import ast
        vibration = ast.literal_eval(row['vibration'])
        
        # Create test data
        test_data = {
            "temp": row['temp'],
            "voltage": row['voltage'],
            "vibration": vibration
        }
        
        # Test prediction
        result = test_single_prediction(test_data, expected_label=1)
        if result:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"  Sample {idx}: temp={row['temp']:.1f}°C, voltage={row['voltage']:.1f}V, "
              f"prediction={'CORRECT' if result else 'WRONG'}")
    
    accuracy = correct_predictions / total_predictions
    print(f"\n=== RESULTS ===")
    print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy > 0.9:
        print("✓ Model is working correctly with training patterns")
    else:
        print("⚠ Model may have issues with training patterns")

def test_single_prediction(test_data, expected_label):
    """Test a single prediction and return if it's correct"""
    
    # Write test data to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        # Run prediction
        result = subprocess.run(['python', 'python_ml/predict_ann.py', temp_file], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            prediction = json.loads(result.stdout)
            predicted_fault = prediction['fault']
            probability = prediction['prob']
            
            # Check if prediction matches expected label
            is_correct = (predicted_fault == expected_label)
            
            print(f"    Predicted: {'FAULT' if predicted_fault else 'NORMAL'} "
                  f"(prob={probability:.3f}), "
                  f"Expected: {'FAULT' if expected_label else 'NORMAL'} -> "
                  f"{'✓' if is_correct else '✗'}")
            
            return is_correct
        else:
            print(f"    Error running prediction: {result.stderr}")
            return False
            
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    return False

if __name__ == "__main__":
    test_model_with_training_patterns()
