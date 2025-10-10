import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from config import RANDOM_SEED, NUM_FAULT_CLASSES

def train_edge_rf_model(X_train, y_train, X_test, y_test, tune=False, rf_n_iter=2, rf_cv=3):
    start = time.time()
    if tune:
        rf = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1)
        param_dist = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=rf_n_iter,
                                    cv=rf_cv, random_state=RANDOM_SEED, n_jobs=-1, verbose=0)
        search.fit(X_train, y_train)
        rf_model = search.best_estimator_
    else:
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_SEED,
                                          class_weight='balanced', n_jobs=-1)
        rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    print(f"RF -> Acc: {acc:.4f}, Macro F1: {f1:.4f} (t={time.time()-start:.2f}s)")
    return rf_model, cm, acc, f1


def _build_dense_model(input_dim, hidden_sizes=(128, 64, 32), dropout=0.2):
    model = Sequential()
    model.add(Dense(hidden_sizes[0], activation='relu', input_shape=(input_dim,)))
    for h in hidden_sizes[1:]:
        model.add(Dropout(dropout))
        model.add(Dense(h, activation='relu'))
    model.add(Dense(NUM_FAULT_CLASSES, activation='softmax'))
    return model


def train_cloud_dense_model(X_train, y_train, X_test, y_test, tune=False, nn_n_iter=2, epochs=5):
    start = time.time()
    y_train_cat = to_categorical(y_train, num_classes=NUM_FAULT_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_FAULT_CLASSES)

    best_model = None
    best_val_loss = np.inf

    if tune:
        rng = np.random.RandomState(RANDOM_SEED)
        hidden_options = [(64, 32), (128, 64), (128, 64, 32)]
        dropout_options = [0.1, 0.2]
        batch_options = [32, 64]
        for i in range(nn_n_iter):
            hs = hidden_options[rng.randint(0, len(hidden_options))]
            dp = float(dropout_options[rng.randint(0, len(dropout_options))])
            bs = int(batch_options[rng.randint(0, len(batch_options))])
            model = _build_dense_model(X_train.shape[1], hidden_sizes=hs, dropout=dp)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            hist = model.fit(X_train, to_categorical(y_train, NUM_FAULT_CLASSES),
                             validation_split=0.15, epochs=epochs, batch_size=bs, callbacks=[early_stop], verbose=0)
            val_loss = min(hist.history.get('val_loss', [np.inf]))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
    else:
        best_model = _build_dense_model(X_train.shape[1])
        best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        best_model.fit(X_train, to_categorical(y_train, NUM_FAULT_CLASSES),
                       validation_split=0.15, epochs=epochs, batch_size=32,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=0)

    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    print(f"Dense NN -> Acc: {acc:.4f}, Macro F1: {f1:.4f} (t={time.time()-start:.2f}s)")
    return best_model, cm, acc, f1
