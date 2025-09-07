import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import sys
import os

if len(sys.argv) > 1:
    accumulated_data = joblib.load(sys.argv[1])
    df = pd.DataFrame(accumulated_data)
else:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "synthetic_data.csv"))

df['vib_mean'] = df['vibration'].apply(lambda x: np.mean(eval(x) if isinstance(x, str) else x))
df['vib_std'] = df['vibration'].apply(lambda x: np.std(eval(x) if isinstance(x, str) else x))
X = df[['vib_mean', 'vib_std', 'temp', 'voltage']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
joblib.dump(rf, 'rf_model.joblib')
print("Trained and saved rf_model.joblib. Accuracy:", rf.score(X_test, y_test))