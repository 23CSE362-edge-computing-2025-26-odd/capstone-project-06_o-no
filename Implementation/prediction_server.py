import socket
import json
import threading
import time
import os
import numpy as np
import joblib
import tensorflow as tf

# Disable GPU for consistency
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class FastPredictionServer:
    def __init__(self, port=12345):
        self.port = port
        self.model = None
        self.scaler = None
        self.imputer = None
        self.config = None
        self.load_model()
        
    def load_model(self):
        """Load model components once at startup"""
        print("Loading model components...")
        start = time.time()
        
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "python_ml", "ann_model.keras")
        scaler_path = os.path.join(script_dir, "python_ml", "ann_scaler.joblib")
        imputer_path = os.path.join(script_dir, "python_ml", "ann_imputer.joblib")
        config_path = os.path.join(script_dir, "python_ml", "ann_config.json")
        
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        load_time = (time.time() - start) * 1000
        print(f"Model loaded in {load_time:.2f}ms")
        
    def extract_vibration_features(self, vibration_series):
        """Fast feature extraction"""
        data = np.asarray(vibration_series, dtype=np.float32)
        mask = ~np.isnan(data)
        
        if not np.any(mask):
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        clean_data = data[mask]
        if len(clean_data) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        features = np.array([
            np.mean(clean_data),
            np.std(clean_data),
            np.max(clean_data),
            np.min(clean_data),
            len(clean_data) / len(data)
        ], dtype=np.float32)
        
        return features
    
    def predict(self, data):
        """Fast prediction with pre-loaded model"""
        start_time = time.time()
        
        try:
            # Extract data
            vibration = data.get('vibration', [0] * 100)
            temp = float(data.get('temp', 50.0))
            voltage = float(data.get('voltage', 220.0))
            
            # Feature extraction
            vib_features = self.extract_vibration_features(vibration)
            features = np.array([[temp, voltage] + vib_features.tolist()], dtype=np.float32)
            
            # Preprocessing and prediction
            features_imputed = self.imputer.transform(features)
            features_scaled = self.scaler.transform(features_imputed)
            prob = float(self.model.predict(features_scaled, verbose=0)[0][0])
            
            threshold = self.config.get('threshold', 0.5)
            fault = 1 if prob > threshold else 0
            
            latency = (time.time() - start_time) * 1000
            
            return {
                "fault": fault,
                "prob": prob,
                "latency_ms": latency,
                "method": "ann_server",
                "temp": temp,
                "voltage": voltage
            }
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                "fault": 0,
                "prob": 0.0,
                "latency_ms": latency,
                "method": "server_error",
                "error": str(e)
            }
    
    def handle_client(self, client_socket):
        """Handle client prediction request"""
        try:
            # Receive data
            data = client_socket.recv(4096).decode('utf-8')
            request = json.loads(data)
            
            # Make prediction
            result = self.predict(request)
            
            # Send response
            response = json.dumps(result)
            client_socket.send(response.encode('utf-8'))
            
        except Exception as e:
            error_response = json.dumps({"error": str(e)})
            client_socket.send(error_response.encode('utf-8'))
        finally:
            client_socket.close()
    
    def start(self):
        """Start the prediction server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', self.port))
        server_socket.listen(5)
        
        print(f"Fast ANN Prediction Server started on port {self.port}")
        print("Waiting for prediction requests...")
        
        try:
            while True:
                client_socket, addr = server_socket.accept()
                # Handle each request in a separate thread for concurrency
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.start()
                
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            server_socket.close()

if __name__ == "__main__":
    server = FastPredictionServer()
    server.start()
