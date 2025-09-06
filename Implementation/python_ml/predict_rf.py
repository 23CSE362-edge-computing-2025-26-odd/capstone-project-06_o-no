import numpy as np
import joblib
import json
import sys

rf = joblib.load('rf_model.joblib')
with open(sys.argv[1], 'r') as f:
    data = json.load(f)
vib = np.array(data['vibration'])
features = np.array([[np.mean(vib), np.std(vib), data['temp'], data['voltage']]])
prob = rf.predict_proba(features)[0][1]
fault = rf.predict(features)[0]
output = {'prob': float(prob), 'fault': int(fault)}
print(json.dumps(output))