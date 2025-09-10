import numpy as np
import pandas as pd
import os

num_samples = 1000
fault_probability = 0.2
vibration_length = 100
script_dir = os.path.dirname(__file__)
output_file = os.path.join(script_dir, "synthetic_data.csv")

np.random.seed(42)

data = {"vibration": [], "temp": [], "voltage": [], "label": []}

for _ in range(num_samples):
    is_fault = np.random.random() < fault_probability
    label = 1 if is_fault else 0

    time = np.linspace(0, 10, vibration_length)
    if is_fault:
        vibration = 2 * np.sin(2 * time) + np.random.normal(0, 0.5, vibration_length)
    else:
        vibration = np.sin(time) + np.random.normal(0, 0.2, vibration_length)

    nan_mask = np.random.random(vibration_length) < 0.05
    vibration[nan_mask] = np.nan
    vibration = vibration.tolist()

    temp = np.random.normal(70 if is_fault else 50, 5 if is_fault else 3)
    if np.random.random() < 0.05: temp = np.nan

    voltage = np.random.normal(250 if is_fault else 220, 10 if is_fault else 5)
    if np.random.random() < 0.05: voltage = np.nan

    data["vibration"].append(vibration)
    data["temp"].append(temp)
    data["voltage"].append(voltage)
    data["label"].append(label)

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(f"Synthetic data generated and saved to {output_file}")
