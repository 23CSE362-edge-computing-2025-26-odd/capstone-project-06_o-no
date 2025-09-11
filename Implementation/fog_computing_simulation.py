#!/usr/bin/env python3
"""
IntelliPdM Fog Computing Simulation - Continuous Monitoring
"""

from config_reader import ConfigReader
import time
import subprocess
import json
import os
import random
import threading
from datetime import datetime

class FogComputingSimulator:
    def __init__(self):
        print(" Initializing Fog Computing Simulator...")
        self.config = ConfigReader()
        print(" Configuration loaded successfully")
        
        self.machines = self.config.get_num_machines()
        self.simulation_time = self.config.get_sim_duration()
        self.monitoring_interval = self.config.get_monitor_interval()
        self.edge_confidence_threshold = self.config.get_edge_threshold()
        self.python_exec = self.config.get_python_executable()
        
        print(f" Configuration:")
        print(f"   Machines: {self.machines}")
        print(f"   Simulation Time: {self.simulation_time}s")
        print(f"   Monitor Interval: {self.monitoring_interval}s")
        print(f"   Edge Threshold: {self.edge_confidence_threshold}")
        
        # Enhanced results tracking
        self.results = {
            'total_predictions': 0,
            'edge_handled': 0,
            'cloud_handled': 0,
            'edge_correct': 0,
            'cloud_correct': 0,
            'edge_latencies': [],
            'cloud_latencies': [],
            'predictions_log': [],
            'machine_stats': {i: {'predictions': 0, 'faults_detected': 0} for i in range(1, self.machines + 1)}
        }
        
        self.running = True
        self.lock = threading.Lock()
    
    def generate_sensor_data(self, machine_id):
        """Generate realistic sensor data with varying fault conditions"""
        # Simulate different machine states with probability
        fault_probability = 0.2  # 20% chance of fault condition
        is_fault = random.random() < fault_probability
        
        if is_fault:
            # Fault conditions
            temp = random.uniform(65, 85)  # High temperature
            voltage = random.uniform(240, 260)  # High voltage
            vibration_x = random.uniform(2.0, 4.0)  # High vibration
            vibration_y = random.uniform(2.0, 4.0)
            vibration_z = random.uniform(2.0, 4.0)
            current = random.uniform(18, 25)  # High current
            true_fault = 1
        else:
            # Normal conditions
            temp = random.uniform(45, 60)
            voltage = random.uniform(210, 235)
            vibration_x = random.uniform(0.5, 1.5)
            vibration_y = random.uniform(0.5, 1.5)
            vibration_z = random.uniform(0.5, 1.5)
            current = random.uniform(12, 17)
            true_fault = 0
        
        return {
            "temp": round(temp, 2),
            "voltage": round(voltage, 2),
            "vibration_x": round(vibration_x, 2),
            "vibration_y": round(vibration_y, 2),
            "vibration_z": round(vibration_z, 2),
            "current": round(current, 2),
            "true_fault": true_fault,
            "timestamp": time.time(),
            "machine_id": machine_id
        }
    
    def predict_edge(self, sensor_data):
        """Perform edge prediction using ANN"""
        machine_id = sensor_data['machine_id']
        temp_file = f"temp_edge_{machine_id}_{int(time.time() * 1000)}.json"
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(sensor_data, f)
            
            start_time = time.time()
            result = subprocess.run([self.python_exec, 'python_ml/predict_ann.py', temp_file], 
                                  capture_output=True, text=True, timeout=5)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            if result.returncode == 0:
                pred = json.loads(result.stdout)
                confidence = abs(pred['prob'] - 0.5) * 2  # Convert to 0-1 scale
                
                return {
                    'fault': pred['fault'],
                    'probability': pred['prob'],
                    'confidence': confidence,
                    'latency_ms': latency,
                    'location': 'edge',
                    'true_fault': sensor_data['true_fault'],
                    'machine_id': machine_id,
                    'timestamp': sensor_data['timestamp']
                }
            else:
                return None
                
        except Exception as e:
            print(f"    Edge prediction error for Machine-{machine_id}: {str(e)[:50]}")
            return None
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def predict_cloud(self, sensor_data):
        """Perform cloud prediction using Random Forest"""
        machine_id = sensor_data['machine_id']
        temp_file = f"temp_cloud_{machine_id}_{int(time.time() * 1000)}.json"
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(sensor_data, f)
            
            start_time = time.time()
            result = subprocess.run([self.python_exec, 'python_ml/predict_rf.py', temp_file], 
                                  capture_output=True, text=True, timeout=10)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            if result.returncode == 0:
                pred = json.loads(result.stdout)
                
                return {
                    'fault': pred['fault'],
                    'probability': pred['prob'],
                    'confidence': 0.9,  # RF generally has high confidence
                    'latency_ms': latency,
                    'location': 'cloud',
                    'true_fault': sensor_data['true_fault'],
                    'machine_id': machine_id,
                    'timestamp': sensor_data['timestamp']
                }
            else:
                return None
                
        except Exception as e:
            print(f"    Cloud prediction error for Machine-{machine_id}: {str(e)[:50]}")
            return None
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def process_prediction(self, sensor_data):
        """Process sensor data through edge-cloud architecture"""
        # Try edge first
        edge_result = self.predict_edge(sensor_data)
        
        if edge_result and edge_result['confidence'] >= self.edge_confidence_threshold:
            # Edge can handle confidently
            final_result = edge_result
            final_result['final_location'] = 'edge'
            
            with self.lock:
                self.results['edge_handled'] += 1
                self.results['edge_latencies'].append(edge_result['latency_ms'])
                if edge_result['fault'] == edge_result['true_fault']:
                    self.results['edge_correct'] += 1
        else:
            # Send to cloud for better accuracy
            cloud_result = self.predict_cloud(sensor_data)
            
            if cloud_result:
                final_result = cloud_result
                final_result['final_location'] = 'cloud'
                final_result['edge_confidence'] = edge_result['confidence'] if edge_result else 0.0
                
                with self.lock:
                    self.results['cloud_handled'] += 1
                    self.results['cloud_latencies'].append(cloud_result['latency_ms'])
                    if cloud_result['fault'] == cloud_result['true_fault']:
                        self.results['cloud_correct'] += 1
            else:
                # Fallback
                final_result = {
                    'fault': 1 if sensor_data['temp'] > 70 else 0,
                    'probability': 0.7,
                    'confidence': 0.5,
                    'latency_ms': 5.0,
                    'location': 'fallback',
                    'final_location': 'fallback',
                    'true_fault': sensor_data['true_fault'],
                    'machine_id': sensor_data['machine_id'],
                    'timestamp': sensor_data['timestamp']
                }
        
        # Update statistics
        with self.lock:
            self.results['total_predictions'] += 1
            self.results['predictions_log'].append(final_result)
            self.results['machine_stats'][sensor_data['machine_id']]['predictions'] += 1
            if final_result['fault'] == 1:
                self.results['machine_stats'][sensor_data['machine_id']]['faults_detected'] += 1
        
        return final_result
    
    def monitor_machine(self, machine_id, start_time):
        """Continuously monitor a single machine"""
        print(f" Starting monitoring thread for Machine-{machine_id}")
        
        while self.running and (time.time() - start_time < self.simulation_time):
            try:
                # Generate sensor data
                sensor_data = self.generate_sensor_data(machine_id)
                
                # Process through fog architecture
                prediction = self.process_prediction(sensor_data)
                
                # Log the result
                fault_status = "FAULT" if prediction['fault'] == 1 else "NORMAL"
                expected = "FAULT" if prediction['true_fault'] == 1 else "NORMAL"
                correct = "" if prediction['fault'] == prediction['true_fault'] else ""
                location_icon = "" if prediction['final_location'] == 'edge' else "" if prediction['final_location'] == 'cloud' else ""
                
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] {location_icon} Machine-{machine_id}: {fault_status} "
                      f"(prob={prediction['probability']:.3f}, {prediction['latency_ms']:.1f}ms) "
                      f"Expected: {expected} {correct}")
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"    Error monitoring Machine-{machine_id}: {str(e)[:50]}")
                time.sleep(self.monitoring_interval)
        
        print(f" Monitoring stopped for Machine-{machine_id}")
    
    def run_simulation(self):
        """Run the complete fog computing simulation"""
        print("=" * 80)
        print(" INTELLIPDM FOG COMPUTING SIMULATION")
        print(" Edge: ANN |  Cloud: Random Forest")
        print("=" * 80)
        
        # Check models
        ann_exists = os.path.exists("python_ml/ann_model.keras")
        rf_exists = os.path.exists("python_ml/rf_model.pkl")
        print(f" Models: ANN {'' if ann_exists else ''}, RF {'' if rf_exists else ''}")
        
        if not ann_exists or not rf_exists:
            print(" Missing models! Please train models first.")
            return
        
        print(f"\n Starting continuous monitoring for {self.simulation_time}s...")
        print(f"   Monitoring {self.machines} machines every {self.monitoring_interval}s")
        print()
        
        start_time = time.time()
        threads = []
        
        try:
            # Start monitoring threads for each machine
            for machine_id in range(1, self.machines + 1):
                thread = threading.Thread(
                    target=self.monitor_machine,
                    args=(machine_id, start_time),
                    name=f"Machine-{machine_id}"
                )
                thread.daemon = True
                thread.start()
                threads.append(thread)
            
            # Monitor simulation progress
            while time.time() - start_time < self.simulation_time:
                elapsed = time.time() - start_time
                remaining = self.simulation_time - elapsed
                
                if int(elapsed) % 30 == 0 and elapsed > 0:  # Progress every 30 seconds
                    with self.lock:
                        progress = (elapsed / self.simulation_time) * 100
                        print(f"\n Progress: {elapsed:.0f}s / {self.simulation_time}s ({progress:.1f}%) - "
                              f"Predictions: {self.results['total_predictions']}")
                
                time.sleep(2)
            
            print(f"\n Simulation time completed, stopping monitoring...")
            self.running = False
            
            # Wait for threads to finish
            for thread in threads:
                thread.join(timeout=3)
                
        except KeyboardInterrupt:
            print("\n Simulation interrupted by user")
            self.running = False
        
        # Display comprehensive results
        self.display_results()
    
    def display_results(self):
        """Display comprehensive simulation results"""
        print("\n" + "=" * 80)
        print(" FOG COMPUTING SIMULATION RESULTS")
        print("=" * 80)
        
        # Overall metrics
        total = self.results['total_predictions']
        overall_correct = self.results['edge_correct'] + self.results['cloud_correct']
        overall_accuracy = (overall_correct / total * 100) if total > 0 else 0
        
        print(f" Overall Performance:")
        print(f"   Total Predictions: {total}")
        print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
        print(f"   Correct Predictions: {overall_correct}")
        print()
        
        # Edge metrics
        if self.results['edge_handled'] > 0:
            edge_accuracy = (self.results['edge_correct'] / self.results['edge_handled'] * 100)
            avg_edge_latency = sum(self.results['edge_latencies']) / len(self.results['edge_latencies'])
            edge_pct = (self.results['edge_handled'] / total * 100)
            
            print(f" Edge Processing (ANN):")
            print(f"   Predictions Handled: {self.results['edge_handled']} ({edge_pct:.1f}%)")
            print(f"   Accuracy: {edge_accuracy:.1f}%")
            print(f"   Average Latency: {avg_edge_latency:.1f}ms")
            print(f"   Min/Max Latency: {min(self.results['edge_latencies']):.1f}ms / {max(self.results['edge_latencies']):.1f}ms")
        else:
            print(f" Edge Processing (ANN): No predictions handled")
        print()
        
        # Cloud metrics
        if self.results['cloud_handled'] > 0:
            cloud_accuracy = (self.results['cloud_correct'] / self.results['cloud_handled'] * 100)
            avg_cloud_latency = sum(self.results['cloud_latencies']) / len(self.results['cloud_latencies'])
            cloud_pct = (self.results['cloud_handled'] / total * 100)
            
            print(f" Cloud Processing (Random Forest):")
            print(f"   Predictions Handled: {self.results['cloud_handled']} ({cloud_pct:.1f}%)")
            print(f"   Accuracy: {cloud_accuracy:.1f}%")
            print(f"   Average Latency: {avg_cloud_latency:.1f}ms")
            print(f"   Min/Max Latency: {min(self.results['cloud_latencies']):.1f}ms / {max(self.results['cloud_latencies']):.1f}ms")
        else:
            print(f" Cloud Processing: No predictions handled")
        print()
        
        # Per-machine statistics
        print(f" Per-Machine Statistics:")
        for machine_id, stats in self.results['machine_stats'].items():
            fault_rate = (stats['faults_detected'] / stats['predictions'] * 100) if stats['predictions'] > 0 else 0
            print(f"   Machine-{machine_id}: {stats['predictions']} predictions, "
                  f"{stats['faults_detected']} faults detected ({fault_rate:.1f}%)")
        print()
        
        # Fog computing benefits
        print(" Fog Computing Benefits Demonstrated:")
        print("    Edge-first processing for low latency")
        print("    Intelligent cloud offloading for complex cases")
        print("    Continuous multi-machine monitoring")
        print("    Real-time fault detection and alerting")
        print("    Fault tolerance with fallback mechanisms")
        print("    Load balancing between edge and cloud")
        
        # Final status
        if overall_accuracy >= 80:
            status = " EXCELLENT"
        elif overall_accuracy >= 70:
            status = " GOOD"
        elif overall_accuracy >= 60:
            status = " ACCEPTABLE"
        else:
            status = " NEEDS IMPROVEMENT"
        
        print(f"\n Simulation Status: {status} ({overall_accuracy:.1f}% accuracy)")
        
        # Save results to file
        self.save_results()
    
    def save_results(self):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fog_simulation_results_{timestamp}.json"
        
        try:
            results_summary = {
                'simulation_config': {
                    'machines': self.machines,
                    'simulation_time': self.simulation_time,
                    'monitoring_interval': self.monitoring_interval,
                    'edge_threshold': self.edge_confidence_threshold
                },
                'overall_stats': {
                    'total_predictions': self.results['total_predictions'],
                    'edge_handled': self.results['edge_handled'],
                    'cloud_handled': self.results['cloud_handled'],
                    'edge_correct': self.results['edge_correct'],
                    'cloud_correct': self.results['cloud_correct'],
                    'overall_accuracy': ((self.results['edge_correct'] + self.results['cloud_correct']) / self.results['total_predictions'] * 100) if self.results['total_predictions'] > 0 else 0
                },
                'latency_stats': {
                    'edge_latencies': self.results['edge_latencies'],
                    'cloud_latencies': self.results['cloud_latencies']
                },
                'machine_stats': self.results['machine_stats'],
                'timestamp': timestamp
            }
            
            with open(filename, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print(f"\n Results saved to: {filename}")
            
        except Exception as e:
            print(f"\n Could not save results: {e}")

if __name__ == "__main__":
    print(" Starting IntelliPdM Fog Computing Simulation...")
    
    try:
        simulator = FogComputingSimulator()
        simulator.run_simulation()
        
    except Exception as e:
        print(f" Simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n Simulation completed!")
