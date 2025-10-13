# models.py (replace previous functions with these)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Optional SMOTE: import if available (used to oversample minority classes)
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

from config import RANDOM_SEED, NUM_FAULT_CLASSES, FAULT_TYPES

# ---------------------------
# Focal loss implementation
# ---------------------------
def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss for multi-class classification (categorical).
    Returns a callable loss function for use in model.compile(...)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # y_true: one-hot, y_pred: probabilities
        # Clip to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss_fn

# ---------------------------
# Random Forest (Edge)
# ---------------------------
def train_edge_rf_model(X_train, y_train, X_test, y_test, use_smote=False):
    """
    Train a RandomForest for edge. Returns (rf_model, confusion_matrix).
    - If use_smote=True and imblearn is installed, SMOTE will be applied to X_train/y_train.
    """
    print("\nTraining Random Forest model for Edge Device...")
    print("-" * 50)

    # Optionally apply SMOTE to training data to reduce class imbalance
    if use_smote and _HAS_SMOTE:
        try:
            sm = SMOTE(random_state=RANDOM_SEED)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            print(f"SMOTE applied: {np.bincount(y_train)} -> {np.bincount(y_train_res)}")
        except Exception as e:
            print("SMOTE failed, falling back to original data:", e)
            X_train_res, y_train_res = X_train, y_train
    else:
        if use_smote and not _HAS_SMOTE:
            print("imblearn not found; install with `pip install imbalanced-learn` to use SMOTE. Proceeding without SMOTE.")
        X_train_res, y_train_res = X_train, y_train

    # Tuned RF hyperparameters
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,                 # let trees grow, but controlled via min_samples_leaf
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight='balanced_subsample',  # balanced per tree
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    rf.fit(X_train_res, y_train_res)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Edge RF Model Performance:")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - Macro F1-Score: {f1:.4f}")
    print("\nClassification Report (Edge Model):")
    print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))

    return rf, confusion_matrix(y_test, y_pred)


# ---------------------------
# Cloud Dense NN (with focal loss + class weights)
# ---------------------------
def train_cloud_dense_model(X_train, y_train, X_test, y_test, use_smote=False):
    """
    Train a cloud-side neural network.
    - Uses dampened class weights and focal loss to improve minority-class learning.
    - Optionally applies SMOTE to X_train/y_train if use_smote=True and imblearn present.
    """
    print("\nTraining Dense Neural Network for Cloud...")
    print("-" * 50)

    # Optional SMOTE application (use with caution for high-dimensional features)
    if use_smote and _HAS_SMOTE:
        try:
            sm = SMOTE(random_state=RANDOM_SEED)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            print(f"SMOTE applied to NN training set: {np.bincount(y_train)} -> {np.bincount(y_train_res)}")
        except Exception as e:
            print("SMOTE failed for NN, using original data:", e)
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train
        if use_smote and not _HAS_SMOTE:
            print("imblearn not present; installing it will enable SMOTE oversampling.")

    # Compute class weights (balanced) then optionally dampen them to avoid overcompensation
    class_weights_raw = compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res)
    # Dampening factor: sqrt reduces extreme weights
    dampening = np.sqrt
    class_weights_dampened = dampening(class_weights_raw)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(np.unique(y_train_res), class_weights_dampened)}

    print("Using class weights (dampened):", class_weight_dict)

    # One-hot encode labels for NN
    y_train_cat = tf.keras.utils.to_categorical(y_train_res, num_classes=NUM_FAULT_CLASSES)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=NUM_FAULT_CLASSES)

    # Build a slightly larger NN with batchnorm and moderate dropout
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.35),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(NUM_FAULT_CLASSES, activation='softmax')
    ])

    # Compile with focal loss and Adam optimizer
    fl = focal_loss(alpha=0.25, gamma=2.0)
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=fl, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0, min_lr=1e-6)
    ]

    # Fit model
    model.fit(
        X_train_res, y_train_cat,
        validation_split=0.15,
        epochs=80,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate and produce classification report
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Cloud Dense Model Performance:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Macro F1-Score: {f1:.4f}")
    print("\nClassification Report (Cloud Model):")
    print(classification_report(y_test, y_pred, target_names=list(FAULT_TYPES.values()), zero_division=0))

    return model, confusion_matrix(y_test, y_pred)
