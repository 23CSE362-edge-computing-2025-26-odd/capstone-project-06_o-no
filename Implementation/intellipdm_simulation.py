#!/usr/bin/env python3
"""
Clean IntelliPdM Simulation - ANN for Edge, Random Forest for Cloud
This script demonstrates the fog computing architecture with:
- ANN for fast edge processing  
- Random Forest for complex cloud processing
- Configuration-driven setup from config.properties
"""

import subprocess
import os
import time
import json
import random
from config_reader import config

class IntelliPdMSimulation:
    def __init__(self):
        # Load configuration from config.properties
        self.machines = config.get_int('numMachines', 5)
        self.time_points = 5  # Fixed for batch simulation
        self.edge_confidence_threshold = config.get_float('loadThreshold', 0.6)
        self.python_exec = config.get('pythonExec', 'python')
        self.project_dir = config.get('projectDir', os.getcwd())
        self.debug = config.get_bool('debug', False)
        
        self.results = {
            'total_predictions': 0,
            'edge_handled': 0,
            'cloud_handled': 0,
            'correct_predictions': 0
        }
    
    def check_models(self):
        """Check if required models exist"""
        print("1. Checking required models...")
        
        ann_model = "python_ml/ann_model.keras"
        rf_model = "python_ml/rf_model.pkl"
        
        if os.path.exists(ann_model):
            print("   ✓ ANN model (Edge) exists")
        else:
            print("   ✗ ANN model missing, training...")
            self.train_ann()
            
        if os.path.exists(rf_model):
            print("   ✓ Random Forest model (Cloud) exists")
        else:
            print("   ✗ Random Forest model missing, training...")
            self.train_rf()
    
    def train_ann(self):
        """Train ANN model for edge processing"""
        try:
            result = subprocess.run([self.python_exec, 'python_ml/train_ann.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✓ ANN model trained successfully")
            else:
                print(f"   ✗ ANN training failed: {result.stderr}")
        except Exception as e:
            print(f"   ✗ Error training ANN: {e}")
    
    def train_rf(self):
        """Train Random Forest model for cloud processing"""
        try:
            result = subprocess.run([self.python_exec, 'python_ml/train_rf.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✓ Random Forest model trained successfully")
            else:
                print(f"   ✗ RF training failed: {result.stderr}")
        except Exception as e:
            print(f"   ✗ Error training RF: {e}")
    
    def generate_sensor_data(self, machine_id, time_point):
        """Generate realistic sensor data"""
        # Generate realistic temperature and voltage ranges
        temp = random.uniform(45, 80)
        voltage = random.uniform(200, 260)
        
        # Generate vibration time series
        vibration = []
        for i in range(100):
            base_val = random.gauss(0, 1.5)
            if random.random() < 0.05:  # 5% chance of spike
                base_val += random.uniform(2, 4)
            vibration.append(base_val)
        
        # Determine expected fault (based on learned patterns)
        expected_fault = 1 if (temp > 60 or voltage > 235) else 0
        
        return {
            "temp": temp,
            "voltage": voltage,
            "vibration": vibration,
            "expected_fault": expected_fault
        }
    
    def edge_prediction(self, sensor_data, machine_id, time_point):
        """Process data at edge using ANN"""
        temp_file = f"temp_edge_{machine_id}_{time_point}.json"
        
        try:
            # Write sensor data to temp file
            with open(temp_file, 'w') as f:
                json.dump(sensor_data, f)
            
            # Run ANN prediction
            result = subprocess.run([self.python_exec, 'python_ml/predict_ann.py', temp_file], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                prediction = json.loads(result.stdout)
                prediction['processing_location'] = 'edge'
                
                # Calculate confidence (distance from 0.5)
                confidence = abs(prediction['prob'] - 0.5) * 2
                prediction['confidence'] = confidence
                
                # Decide if edge can handle this or needs cloud
                if confidence >= self.edge_confidence_threshold:
                    # Edge handles it
                    self.results['edge_handled'] += 1
                    return prediction
                else:
                    # Send to cloud for better accuracy
                    print(f"      ↗️ Low confidence ({confidence:.2f}), sending to cloud...")
                    return self.cloud_prediction(sensor_data, machine_id, time_point)
            else:
                # Edge failed, fallback to cloud
                print(f"      ❌ Edge failed, fallback to cloud...")
                return self.cloud_prediction(sensor_data, machine_id, time_point)
                
        except Exception as e:
            print(f"      ❌ Edge error: {e}, fallback to cloud...")
            return self.cloud_prediction(sensor_data, machine_id, time_point)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def cloud_prediction(self, sensor_data, machine_id, time_point):
        """Process data at cloud using Random Forest"""
        temp_file = f"temp_cloud_{machine_id}_{time_point}.json"
        
        try:
            # Write sensor data to temp file
            with open(temp_file, 'w') as f:
                json.dump(sensor_data, f)
            
            # Run Random Forest prediction
            result = subprocess.run([self.python_exec, 'python_ml/predict_rf.py', temp_file], 
                                  capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                prediction = json.loads(result.stdout)
                prediction['processing_location'] = 'cloud'
                self.results['cloud_handled'] += 1
                return prediction
            else:
                # Both edge and cloud failed, use simple fallback
                print(f"      ❌ Cloud also failed, using fallback...")
                return {
                    'fault': 1 if (sensor_data['temp'] > 65 or sensor_data['voltage'] > 250) else 0,
                    'prob': 0.8,
                    'latency_ms': 5.0,
                    'method': 'fallback',
                    'processing_location': 'fallback'
                }
                
        except Exception as e:
            print(f"      ❌ Cloud error: {e}, using fallback...")
            return {
                'fault': 1 if (sensor_data['temp'] > 65 or sensor_data['voltage'] > 250) else 0,
                'prob': 0.8,
                'latency_ms': 5.0,
                'method': 'fallback',
                'processing_location': 'fallback'
            }
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def run_simulation(self):
        """Run the main simulation"""
        print("=" * 80)
        print("🌐 INTELLIPDM PREDICTIVE MAINTENANCE SIMULATION")
        print("📡 Edge: ANN | ☁️ Cloud: Random Forest")
        print("=" * 80)
        
        # Check and prepare models
        self.check_models()
        
        print(f"\n2. Running simulation...")
        print(f"   Machines: {self.machines}")
        print(f"   Time points per machine: {self.time_points}")
        print(f"   Edge confidence threshold: {self.edge_confidence_threshold}")
        
        print(f"\n3. Processing data...")
        
        # Process each machine
        for machine_id in range(1, self.machines + 1):
            print(f"\n   Machine-{machine_id} Processing:")
            
            for time_point in range(self.time_points):
                # Generate sensor data
                sensor_data = self.generate_sensor_data(machine_id, time_point)
                
                # Process through edge (with potential cloud offload)
                prediction = self.edge_prediction(sensor_data, machine_id, time_point)
                
                if prediction:
                    self.results['total_predictions'] += 1
                    
                    # Check accuracy
                    predicted_fault = prediction['fault']
                    expected_fault = sensor_data['expected_fault']
                    correct = predicted_fault == expected_fault
                    
                    if correct:
                        self.results['correct_predictions'] += 1
                    
                    # Display result
                    status = "FAULT" if predicted_fault == 1 else "NORMAL"
                    expected = "FAULT" if expected_fault == 1 else "NORMAL"
                    accuracy_mark = "✓" if correct else "✗"
                    location_icon = "📱" if prediction['processing_location'] == 'edge' else "☁️"
                    
                    print(f"     {location_icon} Time {time_point}: {status} "
                          f"(prob={prediction['prob']:.3f}, "
                          f"latency={prediction['latency_ms']:.1f}ms, "
                          f"{prediction['processing_location']}) "
                          f"Expected: {expected} {accuracy_mark}")
        
        # Display results
        self.display_results()
    
    def display_results(self):
        """Display simulation results"""
        print(f"\n" + "=" * 80)
        print("📊 SIMULATION RESULTS")
        print("=" * 80)
        
        total = self.results['total_predictions']
        correct = self.results['correct_predictions']
        edge_handled = self.results['edge_handled']
        cloud_handled = self.results['cloud_handled']
        
        overall_accuracy = (correct / total * 100) if total > 0 else 0
        edge_percentage = (edge_handled / total * 100) if total > 0 else 0
        cloud_percentage = (cloud_handled / total * 100) if total > 0 else 0
        
        print(f"📈 Overall Performance:")
        print(f"   Total Predictions: {total}")
        print(f"   Correct Predictions: {correct}")
        print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
        
        print(f"\n📡 Edge Processing (ANN):")
        print(f"   Predictions Handled: {edge_handled} ({edge_percentage:.1f}%)")
        print(f"   Advantage: Low latency, immediate response")
        
        print(f"\n☁️ Cloud Processing (Random Forest):")
        print(f"   Predictions Handled: {cloud_handled} ({cloud_percentage:.1f}%)")
        print(f"   Advantage: Higher accuracy for complex cases")
        
        status = "✓ SUCCESSFUL" if overall_accuracy >= 70 else "⚠ NEEDS IMPROVEMENT"
        print(f"\n🎯 Simulation Status: {status}")
        
        print(f"\n💡 Key Features Demonstrated:")
        print(f"   ✓ Hybrid edge-cloud architecture")
        print(f"   ✓ Intelligent offloading based on confidence")
        print(f"   ✓ ANN for fast edge processing")
        print(f"   ✓ Random Forest for complex cloud analysis")
        print(f"   ✓ Fault tolerance with fallback mechanisms")
        print(f"   ✓ Real-time multi-machine monitoring")

if __name__ == "__main__":
    simulation = IntelliPdMSimulation()
    simulation.run_simulation()
