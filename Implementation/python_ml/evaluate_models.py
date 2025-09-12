#!/usr/bin/env python3
"""
Comprehensive Model Evaluation: ANN vs Random Forest
Get detailed metrics for both models
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from predict_ann import extract_vibration_features

print("=" * 60)
print("INTELLIPDM MODEL EVALUATION REPORT")
print("=" * 60)

# Load data
df = pd.read_csv('synthetic_data.csv')

def safe_parse(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except:
            return []
    return v

df['vibration'] = df['vibration'].apply(safe_parse)

print(f"\nDataset Overview:")
print(f"Total samples: {len(df)}")
print(f"Normal samples: {len(df[df['label'] == 0])}")
print(f"Fault samples: {len(df[df['label'] == 1])}")
print(f"Class distribution: {df['label'].value_counts().values}")

# Prepare ANN features (7 features)
def prepare_ann_features(row):
    vib_features = extract_vibration_features(row['vibration'])
    return [row['temp'], row['voltage']] + vib_features

X_ann = np.array([prepare_ann_features(row) for _, row in df.iterrows()], dtype=np.float32)

# Prepare RF features (3 features)
X_rf = np.array([[np.nanmean(row['vibration']), row['temp'], row['voltage']] for _, row in df.iterrows()])

# Handle missing values for RF
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
X_rf = imputer.fit_transform(X_rf)

y = df['label'].values

# Split data
X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(X_ann, y, test_size=0.2, random_state=42)
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y, test_size=0.2, random_state=42)

print("\n" + "=" * 60)
print("ANN MODEL EVALUATION")
print("=" * 60)

try:
    # Load ANN model and preprocessing
    ann_model = tf.keras.models.load_model("ann_model.keras")
    ann_scaler = joblib.load("ann_scaler.joblib")
    ann_imputer = joblib.load("ann_imputer.joblib")
    
    # Preprocess test data
    X_ann_test_imputed = ann_imputer.transform(X_ann_test)
    X_ann_test_scaled = ann_scaler.transform(X_ann_test_imputed)
    
    # Predictions
    ann_probs = ann_model.predict(X_ann_test_scaled, verbose=0)
    ann_predictions = (ann_probs[:, 0] > 0.5).astype(int)
    
    # Calculate metrics
    ann_accuracy = accuracy_score(y_ann_test, ann_predictions)
    ann_precision = precision_score(y_ann_test, ann_predictions)
    ann_recall = recall_score(y_ann_test, ann_predictions)
    ann_f1 = f1_score(y_ann_test, ann_predictions)
    
    print(f"Architecture: {len(X_ann[0])}-input Neural Network")
    print(f"Features: Temperature, Voltage, Vibration (mean, std, min, max, rms)")
    print(f"Test samples: {len(X_ann_test)}")
    print(f"Accuracy:  {ann_accuracy:.4f} ({ann_accuracy*100:.2f}%)")
    print(f"Precision: {ann_precision:.4f}")
    print(f"Recall:    {ann_recall:.4f}")
    print(f"F1-Score:  {ann_f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm_ann = confusion_matrix(y_ann_test, ann_predictions)
    print(f"[[TN={cm_ann[0,0]}, FP={cm_ann[0,1]}],")
    print(f" [FN={cm_ann[1,0]}, TP={cm_ann[1,1]}]]")
    
    # Calculate confidence statistics
    confidence_scores = np.abs(ann_probs[:, 0] - 0.5) * 2  # Convert to 0-1 scale
    print(f"\nConfidence Analysis:")
    print(f"Average confidence: {np.mean(confidence_scores):.4f}")
    print(f"High confidence (>0.7): {np.sum(confidence_scores > 0.7)} samples ({np.sum(confidence_scores > 0.7)/len(confidence_scores)*100:.1f}%)")
    print(f"Low confidence (<0.7): {np.sum(confidence_scores <= 0.7)} samples ({np.sum(confidence_scores <= 0.7)/len(confidence_scores)*100:.1f}%)")

except Exception as e:
    print(f"Error loading ANN model: {e}")

print("\n" + "=" * 60)
print("RANDOM FOREST MODEL EVALUATION")
print("=" * 60)

try:
    # Load Random Forest model
    rf_model = joblib.load("rf_model.pkl")
    
    # Predictions
    rf_predictions = rf_model.predict(X_rf_test)
    rf_probs = rf_model.predict_proba(X_rf_test)
    
    # Calculate metrics
    rf_accuracy = accuracy_score(y_rf_test, rf_predictions)
    rf_precision = precision_score(y_rf_test, rf_predictions)
    rf_recall = recall_score(y_rf_test, rf_predictions)
    rf_f1 = f1_score(y_rf_test, rf_predictions)
    
    print(f"Architecture: Random Forest with {rf_model.n_estimators} trees")
    print(f"Features: Vibration Mean, Temperature, Voltage")
    print(f"Test samples: {len(X_rf_test)}")
    print(f"Accuracy:  {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    print(f"Precision: {rf_precision:.4f}")
    print(f"Recall:    {rf_recall:.4f}")
    print(f"F1-Score:  {rf_f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm_rf = confusion_matrix(y_rf_test, rf_predictions)
    print(f"[[TN={cm_rf[0,0]}, FP={cm_rf[0,1]}],")
    print(f" [FN={cm_rf[1,0]}, TP={cm_rf[1,1]}]]")
    
    # Feature importance
    feature_names = ['Vibration_Mean', 'Temperature', 'Voltage']
    importances = rf_model.feature_importances_
    print(f"\nFeature Importance:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")

except Exception as e:
    print(f"Error loading Random Forest model: {e}")

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

try:
    print(f"{'Metric':<15} {'ANN':<12} {'Random Forest':<15}")
    print("-" * 45)
    print(f"{'Accuracy':<15} {ann_accuracy:.4f}      {rf_accuracy:.4f}")
    print(f"{'Precision':<15} {ann_precision:.4f}      {rf_precision:.4f}")
    print(f"{'Recall':<15} {ann_recall:.4f}      {rf_recall:.4f}")
    print(f"{'F1-Score':<15} {ann_f1:.4f}      {rf_f1:.4f}")
    
    print(f"\nModel Characteristics:")
    print(f"ANN: {len(X_ann[0])} features, neural network, edge processing")
    print(f"RF:  {len(X_rf[0])} features, ensemble method, cloud processing")
    
except:
    print("Could not complete comparison - some models failed to load")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
