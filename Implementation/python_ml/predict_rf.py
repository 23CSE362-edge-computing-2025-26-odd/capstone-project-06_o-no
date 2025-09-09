# python_ml/predict_rf.py
import sys, os, json
import numpy as np
import joblib

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "rf_model.pkl")

input_file = sys.argv[1]
with open(input_file, 'r') as f:
    data = json.load(f)

vibration = np.array(data.get('vibration', [0]*100), dtype=float)
mean_vib = float(np.nanmean(vibration)) if vibration.size>0 else 0.0
temp = data.get('temp', 50.0)
if temp is None or (isinstance(temp, float) and np.isnan(temp)): temp = 50.0
voltage = data.get('voltage', 220.0)
if voltage is None or (isinstance(voltage, float) and np.isnan(voltage)): voltage = 220.0

X = np.array([[mean_vib, float(temp), float(voltage)]])

model = joblib.load(model_path)
prob = float(model.predict_proba(X)[0][1])
fault = int(model.predict(X)[0])

print(json.dumps({"prob": prob, "fault": int(fault)}), flush=True)
