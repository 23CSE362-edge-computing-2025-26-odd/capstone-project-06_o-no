import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from config import RANDOM_SEED, NUM_FAULT_CLASSES

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


def train_edge_logistic(X_train, y_train, X_test, y_test, tune=False):
    start = time.time()
    model = LogisticRegression(max_iter=300, solver='saga', multi_class='multinomial', random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    print(f"LogisticRegression -> Acc: {acc:.4f}, Macro F1: {f1:.4f} (t={time.time()-start:.2f}s)")
    return model, cm, acc, f1


def train_lightgbm(X_train, y_train, X_test, y_test, n_estimators=100):
    if lgb is None:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")
    start = time.time()
    model = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    print(f"LightGBM -> Acc: {acc:.4f}, Macro F1: {f1:.4f} (t={time.time()-start:.2f}s)")
    return model, cm, acc, f1


def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=100):
    if xgb is None:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    start = time.time()
    model = xgb.XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    print(f"XGBoost -> Acc: {acc:.4f}, Macro F1: {f1:.4f} (t={time.time()-start:.2f}s)")
    return model, cm, acc, f1


def train_edge_ann_model(X_train, y_train, X_test, y_test, tune=False, nn_n_iter=2, epochs=5):
    start = time.time()
    y_train_cat = to_categorical(y_train, num_classes=NUM_FAULT_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_FAULT_CLASSES)

    def build_ann(input_dim, hidden=(64, 32), dropout=0.2):
        model = Sequential()
        model.add(Dense(hidden[0], activation='relu', input_shape=(input_dim,)))
        for h in hidden[1:]:
            model.add(Dropout(dropout))
            model.add(Dense(h, activation='relu'))
        model.add(Dense(NUM_FAULT_CLASSES, activation='softmax'))
        return model

    best_model = None
    best_val_loss = float('inf')
    if tune:
        rng = np.random.RandomState(RANDOM_SEED)
        hidden_options = [(32, 16), (64, 32)]
        dropout_options = [0.1, 0.2]
        batch_options = [32, 64]
        for i in range(nn_n_iter):
            hs = hidden_options[rng.randint(0, len(hidden_options))]
            dp = float(dropout_options[rng.randint(0, len(dropout_options))])
            bs = int(batch_options[rng.randint(0, len(batch_options))])
            model = build_ann(X_train.shape[1], hidden=hs, dropout=dp)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            hist = model.fit(X_train, y_train_cat, validation_split=0.15, epochs=epochs, batch_size=bs, callbacks=[early_stop], verbose=0)
            val_loss = min(hist.history.get('val_loss', [float('inf')]))
            print(f" ANN try {i+1}/{nn_n_iter}: hs={hs} dp={dp} bs={bs} val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
    else:
        best_model = build_ann(X_train.shape[1])
        best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        best_model.fit(X_train, y_train_cat, validation_split=0.15, epochs=epochs, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=0)

    y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    print(f"ANN -> Acc: {acc:.4f}, Macro F1: {f1:.4f} (t={time.time()-start:.2f}s)")
    return best_model, cm, acc, f1


def train_cloud_cnn_model(X_train, y_train, X_test, y_test, tune=False, nn_n_iter=2, epochs=5):
    start = time.time()
    X_train_resh = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_resh = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_train_cat = to_categorical(y_train, num_classes=NUM_FAULT_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_FAULT_CLASSES)

    def build_cnn(filters=32, kernel=2, dense_units=64):
        model = Sequential()
        model.add(Conv1D(filters, kernel_size=kernel, activation='relu', input_shape=(X_train.shape[1], 1), padding='same'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dense(NUM_FAULT_CLASSES, activation='softmax'))
        return model

    best_model = None
    best_val_loss = float('inf')
    if tune:
        rng = np.random.RandomState(RANDOM_SEED)
        filter_options = [16, 32]
        batch_options = [32, 64]
        for i in range(nn_n_iter):
            f = int(filter_options[rng.randint(0, len(filter_options))])
            bs = int(batch_options[rng.randint(0, len(batch_options))])
            model = build_cnn(filters=f, dense_units=64)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            hist = model.fit(X_train_resh, y_train_cat, validation_split=0.15, epochs=epochs, batch_size=bs, callbacks=[early_stop], verbose=0)
            val_loss = min(hist.history.get('val_loss', [float('inf')]))
            print(f" CNN try {i+1}/{nn_n_iter}: filters={f} bs={bs} val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
    else:
        best_model = build_cnn()
        best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        best_model.fit(X_train_resh, y_train_cat, validation_split=0.15, epochs=epochs, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=0)

    y_pred = np.argmax(best_model.predict(X_test_resh, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    print(f"CNN -> Acc: {acc:.4f}, Macro F1: {f1:.4f} (t={time.time()-start:.2f}s)")
    return best_model, cm, acc, f1


def train_cloud_lstm(X_train, y_train, X_test, y_test, tune=False, epochs=5):
    start = time.time()
    Xtr = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    Xte = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    ytr = to_categorical(y_train, num_classes=NUM_FAULT_CLASSES)
    yte = to_categorical(y_test, num_classes=NUM_FAULT_CLASSES)

    model = Sequential([LSTM(64, input_shape=(Xtr.shape[1], Xtr.shape[2])), Dense(64, activation='relu'), Dense(NUM_FAULT_CLASSES, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_split=0.15, epochs=epochs, batch_size=32, callbacks=[early_stop], verbose=0)
    y_pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    print(f"LSTM -> Acc: {acc:.4f}, Macro F1: {f1:.4f} (t={time.time()-start:.2f}s)")
    return model, cm, acc, f1
