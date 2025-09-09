# python_ml/train_cnn.py
import os
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, "synthetic_data.csv")

df = pd.read_csv(csv_path)

# Safe parser for vibration column
def safe_parse(v):
    if isinstance(v, str):
        try:
            if v.lower() == "nan":
                return np.nan
            return ast.literal_eval(v)  # safe alternative to eval
        except:
            return np.nan
    return v

df['vibration'] = df['vibration'].apply(safe_parse)

# Drop rows where vibration is NaN
df = df.dropna(subset=['vibration'])

# Convert to numpy
X = np.array([v for v in df['vibration']]).reshape(-1, 100, 1).astype('float32')
y = df['label'].values

# Fill NaNs in X with column means
inds = np.where(np.isnan(X))
if len(inds[0]) > 0:
    col_means = np.nanmean(X, axis=0)  # shape (100,1)
    for i in range(X.shape[0]):
        mask = np.isnan(X[i, :, 0])
        X[i, mask, 0] = col_means[mask, 0]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(100, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train (few epochs for demo)
model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

# Save model
model_path = os.path.join(script_dir, "cnn_model.h5")
model.save(model_path)
print("CNN model trained and saved to", model_path)
