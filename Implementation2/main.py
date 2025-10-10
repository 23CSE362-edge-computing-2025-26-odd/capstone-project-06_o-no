import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
import simpy
warnings.filterwarnings('ignore')
from config import RANDOM_SEED, SIMULATION_TIME, EDGE_CAPACITY, CLOUD_CAPACITY, FEATURE_COLUMNS, FAULT_TYPES
from logger import SystemLogger
from data_prep import load_and_prepare_dataset, generate_runtime_sensor_data
from models import train_edge_rf_model, train_cloud_dense_model
from additional_models import (
    train_edge_ann_model,
    train_cloud_cnn_model,
    train_edge_logistic,
    train_lightgbm,
    train_xgboost,
    train_cloud_lstm
)
from ga_scheduler import EnhancedGA
from sim_processes import sensor_process, model_predict_label

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

metrics = {
    'total_tasks': 0, 'total_faults_detected': 0,
    'edge': {'tasks_processed': 0, 'faults_detected': 0, 'latency': [], 'energy': [], 'accuracy': [], 'processing_times': []},
    'cloud': {'tasks_processed': 0, 'faults_detected': 0, 'latency': [], 'energy': [], 'accuracy': [], 'network_latency': [], 'processing_times': []},
    'scheduling': {'offload_decisions': [], 'ga_fitness_scores': [], 'edge_load_history': []},
    'timeline': [], 'logs': []
}

logger = SystemLogger(metrics)

ENABLE_SMOTE = True     
ENABLE_TUNING = True   
RF_N_ITER = 2
NN_N_ITER = 2
EPOCHS = 5

def measure_inference_latency(model, X_test, n_samples=100, warmup=3):
    """
    Measure per-sample inference latency (ms) for `model` on X_test.
    - Uses model_predict_label to mirror simulator's prediction behavior.
    - Runs `warmup` predictions, then times `n_samples` single-sample calls.
    - Returns (mean_ms, std_ms).
    """
    import numpy as _np
    if X_test is None or len(X_test) == 0:
        return float('nan'), float('nan')

    n_samples = int(min(max(1, n_samples), X_test.shape[0]))
    for _ in range(warmup):
        try:
            model_predict_label(model, X_test[0:1])
        except Exception:
            try:
                if hasattr(model, "predict_proba"):
                    _ = model.predict_proba(X_test[0:1])
                else:
                    _ = model.predict(X_test[0:1], verbose=0)
            except Exception:
                pass

    times = []
    for i in range(n_samples):
        x = X_test[i:i+1]
        t0 = time.time()
        try:
            model_predict_label(model, x)
        except Exception:
            try:
                if hasattr(model, "predict_proba"):
                    _ = model.predict_proba(x)
                else:
                    _ = model.predict(x, verbose=0)
            except Exception:
                try:
                    x3 = x.reshape((x.shape[0], x.shape[1], 1))
                    _ = model.predict(x3, verbose=0)
                except Exception:
                    pass
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    times = _np.array(times, dtype=float)
    if times.size == 0:
        return float('nan'), float('nan')
    return float(times.mean()), float(times.std(ddof=0))


def run_full_pipeline():
    """Train models, run simulation (using selected edge and cloud models),
       return metrics, model_summary and scaler used."""
    print("\n" + "="*80)
    print(" " * 12 + "EDGE-CLOUD PREDICTIVE MAINTENANCE - RUN")
    print("="*80 + "\n")

    print("[1] Loading dataset...")
    df = load_and_prepare_dataset()
    print(f" - data shape: {df.shape}")
    print(" - class counts:\n", df['Fault_Label'].value_counts().sort_index())

    X = df[FEATURE_COLUMNS]
    y = df['Fault_Label']

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE optional
    if ENABLE_SMOTE:
        if not HAS_SMOTE:
            print("WARNING: SMOTE requested but imbalanced-learn not installed — skipping SMOTE.")
            X_train_res, y_train_res = X_train_scaled, y_train
        else:
            print("Applying SMOTE to training set (fast).")
            sm = SMOTE(random_state=RANDOM_SEED)
            X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
            print(" - after SMOTE class counts:", np.bincount(y_train_res))
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    print("\n[2] Training models (small budgets/tuning)...")
    rf_model, rf_cm, rf_acc, rf_f1 = train_edge_rf_model(X_train_res, y_train_res, X_test_scaled, y_test,
                                                         tune=ENABLE_TUNING, rf_n_iter=RF_N_ITER)
    dense_model, dense_cm, dense_acc, dense_f1 = train_cloud_dense_model(X_train_res, y_train_res, X_test_scaled, y_test,
                                                                          tune=ENABLE_TUNING, nn_n_iter=NN_N_ITER, epochs=EPOCHS)

    ann_model, ann_cm, ann_acc, ann_f1 = train_edge_ann_model(X_train_res, y_train_res, X_test_scaled, y_test,
                                                              tune=ENABLE_TUNING, nn_n_iter=NN_N_ITER, epochs=EPOCHS)
    cnn_model, cnn_cm, cnn_acc, cnn_f1 = train_cloud_cnn_model(X_train_res, y_train_res, X_test_scaled, y_test,
                                                              tune=ENABLE_TUNING, nn_n_iter=NN_N_ITER, epochs=EPOCHS)

    extras = {}
    try:
        log_model, log_cm, log_acc, log_f1 = train_edge_logistic(X_train_res, y_train_res, X_test_scaled, y_test)
        extras['Logistic'] = {'cm': log_cm, 'acc': log_acc, 'f1': log_f1}
    except Exception as e:
        print(" - Logistic skipped:", e)
    try:
        lgb_model, lgb_cm, lgb_acc, lgb_f1 = train_lightgbm(X_train_res, y_train_res, X_test_scaled, y_test)
        extras['LightGBM'] = {'cm': lgb_cm, 'acc': lgb_acc, 'f1': lgb_f1}
    except Exception as e:
        print(" - LightGBM skipped:", e)
        lgb_model = None
    try:
        xgb_model, xgb_cm, xgb_acc, xgb_f1 = train_xgboost(X_train_res, y_train_res, X_test_scaled, y_test)
        extras['XGBoost'] = {'cm': xgb_cm, 'acc': xgb_acc, 'f1': xgb_f1}
    except Exception as e:
        print(" - XGBoost skipped:", e)
        xgb_model = None
    try:
        lstm_model, lstm_cm, lstm_acc, lstm_f1 = train_cloud_lstm(X_train_res, y_train_res, X_test_scaled, y_test, epochs=EPOCHS)
        extras['LSTM'] = {'cm': lstm_cm, 'acc': lstm_acc, 'f1': lstm_f1}
    except Exception as e:
        print(" - LSTM skipped:", e)
        lstm_model = None

    model_summary = {
        'RF (Edge)': {'acc': rf_acc, 'f1': rf_f1, 'cm': rf_cm},
        'Dense NN (Cloud)': {'acc': dense_acc, 'f1': dense_f1, 'cm': dense_cm},
        'ANN (Edge)': {'acc': ann_acc, 'f1': ann_f1, 'cm': ann_cm},
        'CNN (Cloud)': {'acc': cnn_acc, 'f1': cnn_f1, 'cm': cnn_cm},
    }
    for k, v in extras.items():
        model_summary[k] = {'acc': v['acc'], 'f1': v['f1'], 'cm': v['cm']}

    model_objects = {
        'RF (Edge)': rf_model,
        'Dense NN (Cloud)': dense_model,
        'ANN (Edge)': ann_model,
        'CNN (Cloud)': cnn_model,
    }
    if 'Logistic' in extras:
        model_objects['Logistic'] = locals().get('log_model', None)
    if 'LightGBM' in extras:
        model_objects['LightGBM'] = locals().get('lgb_model', None)
    if 'XGBoost' in extras:
        model_objects['XGBoost'] = locals().get('xgb_model', None)
    if 'LSTM' in extras:
        model_objects['LSTM'] = locals().get('lstm_model', None)

    for name, mdl in model_objects.items():
        if mdl is None:
            if name in model_summary:
                model_summary[name]['latency_ms'] = None
                model_summary[name]['latency_std_ms'] = None
            continue
        mean_lat, std_lat = measure_inference_latency(mdl, X_test_scaled, n_samples=min(100, X_test_scaled.shape[0]))
        if name in model_summary:
            model_summary[name]['latency_ms'] = mean_lat
            model_summary[name]['latency_std_ms'] = std_lat
        else:
            model_summary[name] = {'acc': None, 'f1': None, 'cm': None, 'latency_ms': mean_lat, 'latency_std_ms': std_lat}
        print(f"{name} -> Avg inference latency: {mean_lat:.3f} ms (std {std_lat:.3f} ms)")
   
    edge_preference = ['LightGBM', 'XGBoost', 'RF (Edge)', 'Logistic']
    edge_model = None
    edge_model_name = None
    for nm in edge_preference:
        if nm in model_objects and model_objects[nm] is not None:
            edge_model = model_objects[nm]
            edge_model_name = nm
            break
    if edge_model is None:
        edge_model = rf_model
        edge_model_name = 'RF (Edge)'
    cloud_preference = ['LSTM', 'Dense NN (Cloud)']
    cloud_model = None
    cloud_model_name = None
    for nm in cloud_preference:
        if nm in model_objects and model_objects[nm] is not None:
            cloud_model = model_objects[nm]
            cloud_model_name = nm
            break
    if cloud_model is None:
        cloud_model = dense_model
        cloud_model_name = 'Dense NN (Cloud)'

    print(f"\nSelected deployment models -> EDGE: {edge_model_name}; CLOUD: {cloud_model_name}\n")

    print("\n[3] Starting simulation ...")
    env = simpy.Environment()
    edge_res = simpy.Resource(env, capacity=EDGE_CAPACITY)
    cloud_res = simpy.Resource(env, capacity=CLOUD_CAPACITY)
    ga = EnhancedGA()
    env.process(sensor_process(env, edge_res, cloud_res, ga, metrics, scaler, edge_model, cloud_model, logger, []))
    env.run(until=SIMULATION_TIME)
    print("Simulation finished.")

    return metrics, model_summary, scaler

def compute_simulation_accuracy(metrics):
    edge_acc_list = metrics['edge']['accuracy'] or []
    cloud_acc_list = metrics['cloud']['accuracy'] or []
    total_correct = sum(edge_acc_list) + sum(cloud_acc_list)
    total_processed = len(edge_acc_list) + len(cloud_acc_list)
    if total_processed == 0:
        return 0.0, 0, 0
    return total_correct / total_processed, int(total_correct), total_processed

def per_class_f1_from_cm(cm):
    """Compute per-class F1 array from confusion matrix (cm must be square)."""
    cm = np.array(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_tot = cm.sum(axis=0)
    precision = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return f1, precision, recall, support.astype(int)

def print_and_plot_results(metrics, model_summary):
    print("\n" + "="*60)
    print("SIMULATION & MODEL SUMMARY")
    print("="*60)

    sim_acc, correct, processed = compute_simulation_accuracy(metrics)
    print(f"Simulation accuracy (runtime): {sim_acc*100:.2f}%  ({correct}/{processed} correct)")

    edge_stats = {
        'Tasks': metrics['edge']['tasks_processed'],
        'Avg Latency': np.mean(metrics['edge']['latency'] or [0]),
        'Avg Energy': np.mean(metrics['edge']['energy'] or [0]),
        'Accuracy': np.mean(metrics['edge']['accuracy'] or [0])
    }
    cloud_stats = {
        'Tasks': metrics['cloud']['tasks_processed'],
        'Avg Latency': np.mean(metrics['cloud']['latency'] or [0]),
        'Avg Energy': np.mean(metrics['cloud']['energy'] or [0]),
        'Accuracy': np.mean(metrics['cloud']['accuracy'] or [0])
    }
    print("\nEdge vs Cloud (summary):")
    print(pd.DataFrame({'Edge': edge_stats, 'Cloud': cloud_stats}).round(4))

    print("\nModel comparison (test set):")
    rows = []
    for name, info in model_summary.items():
        rows.append({
            'Model': name,
            'Accuracy': info.get('acc', float('nan')),
            'Macro F1': info.get('f1', float('nan')),
            'Latency_ms': info.get('latency_ms', float('nan'))
        })
    comp_df = pd.DataFrame(rows).reset_index(drop=True)

    comp_df = comp_df.sort_values(by='Accuracy', ascending=False, na_position='last').reset_index(drop=True)

    import numpy as _np
    acc_vals = comp_df['Accuracy'].to_numpy(dtype=float)
    if _np.nanstd(acc_vals) > 0:
        comp_df['Acc_z'] = (acc_vals - _np.nanmean(acc_vals)) / _np.nanstd(acc_vals)
    else:
        comp_df['Acc_z'] = 0.0

    lat_vals = comp_df['Latency_ms'].to_numpy(dtype=float)
    valid_lat = lat_vals[~_np.isnan(lat_vals)]
    if valid_lat.size > 1 and _np.nanstd(valid_lat) > 0:
        mean_lat = _np.nanmean(lat_vals)
        std_lat = _np.nanstd(lat_vals)
        comp_df['Lat_z'] = -1.0 * (lat_vals - mean_lat) / std_lat
    else:
        comp_df['Lat_z'] = 0.0

    with pd.option_context('display.float_format', '{:0.4f}'.format):
        print(comp_df.to_string(index=False))

    plt.figure(figsize=(10,5))
    x = np.arange(len(comp_df))
    w = 0.35
    plt.bar(x - w/2, comp_df['Accuracy'], width=w, label='Accuracy')
    plt.bar(x + w/2, comp_df['Macro F1'], width=w, label='Macro F1')
    plt.xticks(x, comp_df['Model'], rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.title('Model test-set Accuracy and Macro F1')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    if comp_df['Latency_ms'].notna().any():
        plt.bar(comp_df['Model'], comp_df['Latency_ms'])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Avg inference latency (ms)')
        plt.title('Per-model average single-sample inference latency (ms)')
        for i, v in enumerate(comp_df['Latency_ms'].fillna(0)):
            plt.text(i, v + max(1.0, 0.01 * (np.nanmax(comp_df['Latency_ms']) if comp_df['Latency_ms'].notna().any() else 1.0)), f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo latency measurements available to plot.")

    core_models = ['RF (Edge)', 'ANN (Edge)', 'Dense NN (Cloud)', 'CNN (Cloud)']
    core_models += [m for m in model_summary.keys() if m not in core_models]
    labels = list(FAULT_TYPES.values())
    n = len(core_models)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*6, rows*3.5))
    for i, mname in enumerate(core_models):
        if mname not in model_summary:
            continue
        cm = model_summary[mname]['cm']
        if cm is None:
            continue
        f1, prec, rec, supp = per_class_f1_from_cm(cm)
        ax = plt.subplot(rows, cols, i+1)
        ax.barh(labels, f1, height=0.6)
        ax.set_xlim(0,1.0)
        ax.set_xlabel('F1 score')
        ax.set_title(f'Per-class F1 — {mname}')
        for j, v in enumerate(f1):
            ax.text(v + 0.01, j, f"{v:.2f}", va='center')
    plt.tight_layout()
    plt.show()

   
    timeline_df = pd.DataFrame(metrics.get('timeline', []))
    if not timeline_df.empty:
        timeline_df['time'] = pd.to_numeric(timeline_df['time'], errors='coerce')
        timeline_df['latency'] = pd.to_numeric(timeline_df['latency'], errors='coerce')
        edge_df = timeline_df[timeline_df['location'] == 'edge']
        cloud_df = timeline_df[timeline_df['location'] == 'cloud']
    else:
        edge_df = pd.DataFrame(); cloud_df = pd.DataFrame()

    plt.figure(figsize=(10,4))
    if not timeline_df.empty:
        if not edge_df.empty:
            plt.scatter(edge_df['time'], edge_df['latency'], label='Edge', alpha=0.6, s=18)
        if not cloud_df.empty:
            plt.scatter(cloud_df['time'], cloud_df['latency'], label='Cloud', alpha=0.6, s=18)
        plt.xlabel('Simulation time'); plt.ylabel('Latency (ms)')
        plt.title('Latency per processed task over simulation time')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo timeline data available for latency-over-time plot.")

    plt.figure(figsize=(10,4))
    edge_lat = (edge_df['latency'].dropna().tolist() if not edge_df.empty else [])
    cloud_lat = (cloud_df['latency'].dropna().tolist() if not cloud_df.empty else [])
    if len(edge_lat) + len(cloud_lat) > 0:
        all_lat = np.array(edge_lat + cloud_lat)
        max_lat = max(all_lat.max(), 1.0)
        bins = np.linspace(0, max_lat, 11)
        edge_counts, _ = np.histogram(edge_lat, bins=bins)
        cloud_counts, _ = np.histogram(cloud_lat, bins=bins)
        x = np.arange(len(bins)-1)
        width = 0.35
        plt.bar(x - width/2, edge_counts, width=width, label='Edge')
        plt.bar(x + width/2, cloud_counts, width=width, label='Cloud')
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        plt.xticks(x, bin_labels, rotation=45, ha='right', fontsize=8)
        plt.ylabel('Number of tasks'); plt.title('Number of tasks by latency bins (Edge vs Cloud)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo latency data available for histogram.")
    if not timeline_df.empty:
        nbins = 20
        tmin = timeline_df['time'].min()
        tmax = timeline_df['time'].max()
        if tmax > tmin:
            bins = np.linspace(tmin, tmax, nbins+1)
            edge_counts_time, _ = np.histogram(edge_df['time'], bins=bins) if not edge_df.empty else np.zeros(nbins, dtype=int)
            cloud_counts_time, _ = np.histogram(cloud_df['time'], bins=bins) if not cloud_df.empty else np.zeros(nbins, dtype=int)
            mid = (bins[:-1] + bins[1:]) / 2
            plt.figure(figsize=(10,4))
            plt.plot(mid, edge_counts_time.cumsum(), label='Edge (cumulative)', marker='o')
            plt.plot(mid, cloud_counts_time.cumsum(), label='Cloud (cumulative)', marker='o')
            plt.xlabel('Simulation time'); plt.ylabel('Cumulative tasks processed')
            plt.title('Cumulative processed tasks over time (Edge vs Cloud)')
            plt.legend()
            plt.tight_layout()
            plt.show()

    print("\nEnd of results.")


if __name__ == "__main__":
    start = time.time()
    metrics_out, model_summary_out, scaler_used = run_full_pipeline()
    print_and_plot_results(metrics_out, model_summary_out)
    print(f"\nTotal elapsed time: {time.time() - start:.1f}s")
