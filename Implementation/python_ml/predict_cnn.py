# python_ml/predict_cnn.py
import sys, os, json
import numpy as np
from tensorflow.keras.models import load_model

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "model.h5")

input_file = sys.argv[1]
with open(input_file, 'r') as f:
    data = json.load(f)

vibration = np.array(data.get('vibration', [0]*100)).reshape(1, 100, 1).astype('float32')

# handle NaNs by replacing with 0 (or better, mean) for predict
if np.isnan(vibration).any():
    nan_mask = np.isnan(vibration)
    vibration[nan_mask] = 0.0

model = load_model(model_path)
pred = model.predict(vibration)
prob = float(pred[0][0])
fault = 1 if prob > 0.5 else 0

print(json.dumps({"prob": prob, "fault": fault}), flush=True)
