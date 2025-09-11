import pandas as pd
import json
import numpy as np

# Load the data
df = pd.read_csv('synthetic_data.csv')
print(f'Total rows: {len(df)}')
print('Label distribution:', df['label'].value_counts().to_dict())

# Check for complete vibration data
valid_rows = 0
fault_rows_with_valid_data = 0
for i, row in df.iterrows():
    try:
        vib_data = json.loads(row['vibration'])
        vib_array = np.array(vib_data, dtype=float)
        if not np.isnan(vib_array).any() and len(vib_array) == 100:
            valid_rows += 1
            if row['label'] == 1:
                fault_rows_with_valid_data += 1
    except:
        pass

print(f'Rows with complete vibration data: {valid_rows}')
print(f'Fault rows with complete vibration data: {fault_rows_with_valid_data}')

# Check missing temperature and voltage
print(f'Missing temperature: {df["temp"].isna().sum()}')
print(f'Missing voltage: {df["voltage"].isna().sum()}')

# Check a few vibration entries for NaN values
print("\nChecking first few vibration arrays for NaN values:")
for i in range(min(5, len(df))):
    try:
        vib_data = json.loads(df.iloc[i]['vibration'])
        vib_array = np.array(vib_data, dtype=float)
        nan_count = np.isnan(vib_array).sum()
        print(f'Row {i}: {len(vib_array)} values, {nan_count} NaN values, label={df.iloc[i]["label"]}')
    except Exception as e:
        print(f'Row {i}: Error parsing - {e}')
