import numpy as np
from keras.models import load_model
import json
import sys

model = load_model('model.h5')
with open(sys.argv[1], 'r') as f:
    data = json.load(f)
vib = np.array(data['vibration']).reshape(1, 100, 1)
temp = np.array([data['temp']])
volt = np.array([data['voltage']])
prob = model.predict([vib, temp, volt])[0][0]
fault = 1 if prob > 0.5 else 0
output = {'prob': float(prob), 'fault': fault}
print(json.dumps(output))