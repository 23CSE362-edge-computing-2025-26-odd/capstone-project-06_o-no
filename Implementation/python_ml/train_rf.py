import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import ast

csv_file = os.path.join(os.path.dirname(__file__), "synthetic_data.csv")
df = pd.read_csv(csv_file)

def safe_parse(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except:
            return []
    return v

df['vibration'] = df['vibration'].apply(safe_parse)

X = np.array([[np.nanmean(row['vibration']), row['temp'], row['voltage']] for _, row in df.iterrows()])
y = df['label'].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Random Forest Accuracy: {acc:.4f}")

output_file = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
joblib.dump(model, output_file)
print(f"RF model trained and saved to {output_file}")
