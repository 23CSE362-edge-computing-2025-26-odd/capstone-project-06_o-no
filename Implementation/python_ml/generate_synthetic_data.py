import numpy as np
import pandas as pd
import json

def generate_sample(is_fault):
    time = np.linspace(0, 10, 100)
    if is_fault:
        vibration = 2 * np.sin(2 * time) + np.random.normal(0, 0.5, 100)  # Higher freq/amplitude for fault
        temp = np.random.normal(70, 5)  # Higher temp
        voltage = np.random.normal(250, 10)  # Abnormal voltage
    else:
        vibration = np.sin(time) + np.random.normal(0, 0.2, 100)
        temp = np.random.normal(50, 3)
        voltage = np.random.normal(220, 5)
    return vibration, temp, voltage, 1 if is_fault else 0

# Generate 2000 samples (half fault)
data = []
for i in range(2000):
    is_fault = i % 2 == 0
    vib, t, v, label = generate_sample(is_fault)
    data.append({'vibration': vib.tolist(), 'temp': t, 'voltage': v, 'label': label})

df = pd.DataFrame(data)
df.to_csv('synthetic_data.csv', index=False)
print("Generated synthetic_data.csv")