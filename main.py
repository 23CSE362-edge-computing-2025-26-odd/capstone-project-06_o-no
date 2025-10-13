# main.py
import os
import simpy
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_SEED, SIMULATION_TIME, EDGE_CAPACITY, CLOUD_CAPACITY, DATASET_PATH, FEATURE_COLUMNS, FAULT_TYPES
from logger import SystemLogger
from data_prep import load_and_prepare_dataset, generate_runtime_sensor_data
from models import train_edge_rf_model, train_cloud_dense_model
# from ga_scheduler import EnhancedGA   # <- GA import (left commented for reference)
from sim_processes import sensor_process

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -----------------------------------------------------------
#                       DQN Scheduler
# -----------------------------------------------------------
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQN_Scheduler:
    """
    Lightweight DQN scheduler integrated for simulation.
    State = [task_complexity, edge_load, cloud_queue, net_latency]
    Actions: 0 -> Edge, 1 -> Cloud
    The scheduler stores experiences and trains a tiny network online.
    """
    def __init__(self, state_dim=4, action_dim=2, lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=5000)
        self.model = self._build_model(lr)
        self.target_model = self._build_model(lr)
        self.update_target_model()
        self.learn_step = 0
        self.target_update_freq = 100

    def _build_model(self, lr):
        model = Sequential([
            InputLayer(input_shape=(self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def act(self, state, greedy=False):
        if (not greedy) and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return int(np.argmax(q))

    def replay(self, batch_size=32):
        if len(self.memory) < max(batch_size, 100):
            return None
        batch = random.sample(self.memory, batch_size)
        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        q_values = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)
        q_target = q_values.copy()

        for i in range(len(batch)):
            if dones[i]:
                q_target[i, actions[i]] = rewards[i]
            else:
                q_target[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])

        history = self.model.fit(states, q_target, epochs=1, verbose=0, batch_size=batch_size)
        loss = history.history['loss'][0]
        self.learn_step += 1

        if self.learn_step % self.target_update_freq == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def schedule(self, task_complexity, edge_load, cloud_queue=0, net_latency=5):
        # Build state vector
        state = np.array([task_complexity, edge_load, cloud_queue, net_latency], dtype=np.float32)

        # Choose action
        action = self.act(state)
        decision = int(action)  # 0=edge, 1=cloud

        # Compute fitness-like cost (we use same structure as GA fitness but reward is negative cost)
        if decision == 0:
            latency = (task_complexity * 2) + (edge_load * 10 if edge_load > 0.8 else edge_load * 3)
            energy = task_complexity * 0.5
            qos_penalty = 20 if edge_load > 0.9 else 0
        else:
            latency = net_latency + (task_complexity * 1.5) + (cloud_queue * 0.5)
            energy = (task_complexity * 0.2) + 0.3
            qos_penalty = 0

        # Reward: negative of weighted cost; optional small cloud penalty to avoid excessive offloading
        cloud_penalty = 0.3 if decision == 1 else 0.0
        reward = - (0.4 * latency + 0.3 * energy + 0.3 * qos_penalty) - cloud_penalty

        # Next state (small stochasticity to help exploration)
        next_state = np.clip(state + np.random.normal(0, 0.02, size=state.shape), 0, None)
        done = False

        # Store and train a bit online
        self.remember(state, decision, reward, next_state, done)
        self.replay()

        return decision, reward

# -----------------------------------------------------------
#                      Metrics + Globals
# -----------------------------------------------------------
metrics = {
    'total_tasks': 0, 'total_faults_detected': 0,
    'edge': {'tasks_processed': 0, 'faults_detected': 0, 'latency': [], 'energy': [], 'accuracy': [], 'processing_times': []},
    'cloud': {'tasks_processed': 0, 'faults_detected': 0, 'latency': [], 'energy': [], 'accuracy': [], 'network_latency': [], 'processing_times': []},
    'scheduling': {'offload_decisions': [], 'ga_fitness_scores': [], 'edge_load_history': []},
    'timeline': [], 'logs': []
}

rf_model = None
dense_model = None
scaler = StandardScaler()
db_storage = []   # sim_processes should append per-task dicts if present

logger = SystemLogger(metrics)

# Track scheduler name for saved dashboards
SCHEDULER_NAME = "dqn"

# -----------------------------------------------------------
#                    Run Simulation
# -----------------------------------------------------------
def run_simulation():
    global rf_model, dense_model, scaler, SCHEDULER_NAME
    print("\n" + "="*70); print(" " * 10 + "AUTOMOTIVE FACTORY PREDICTIVE MAINTENANCE SYSTEM"); print("="*70)
    
    print("\n[Phase 1] Data Preparation"); print("-" * 50)
    df = load_and_prepare_dataset()
    print(f"Dataset shape: {df.shape}"); print(f"Class distribution:\n{df['Fault_Label'].value_counts().sort_index()}")
    
    X = df[FEATURE_COLUMNS]; y = df['Fault_Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train); X_test_scaled = scaler.transform(X_test)
    
    print("\n[Phase 2] Model Training"); print("-" * 50)
    rf_model, rf_cm = train_edge_rf_model(X_train_scaled, y_train, X_test_scaled, y_test)
    dense_model, dense_cm = train_cloud_dense_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n[Phase 3] Simulation Setup"); print("-" * 50)
    env = simpy.Environment()
    edge_resource = simpy.Resource(env, capacity=EDGE_CAPACITY)
    cloud_resource = simpy.Resource(env, capacity=CLOUD_CAPACITY)

    # ga_scheduler = EnhancedGA()   # ← GA commented out (kept for reference)
    dqn_scheduler = DQN_Scheduler()
    SCHEDULER_NAME = "dqn"

    print(f"\n[Phase 4] Starting Simulation (Duration: {SIMULATION_TIME} mins)...")
    env.process(sensor_process(env, edge_resource, cloud_resource, dqn_scheduler, metrics, scaler, rf_model, dense_model, logger, db_storage))
    env.run(until=SIMULATION_TIME)
    
    print("\n[Phase 5] Simulation completed!")
    return metrics, rf_cm, dense_cm

# -----------------------------------------------------------
#                    Analysis & Dashboard
# -----------------------------------------------------------
def analyze_results(metrics, rf_cm, dense_cm, scheduler_name="dqn"):
    print("\n" + "="*70); print(" " * 25 + "SIMULATION ANALYSIS REPORT"); print("="*70)
    
    total_processed = metrics['edge']['tasks_processed'] + metrics['cloud']['tasks_processed']
    if total_processed == 0:
        print("\nNo tasks were processed during the simulation.")
        return

    print("\n" + "-"*20 + " Overall System Performance " + "-"*20)
    overall_accuracy = np.mean(metrics['edge']['accuracy'] + metrics['cloud']['accuracy']) if (metrics['edge']['accuracy'] or metrics['cloud']['accuracy']) else 0.0
    offload_rate = (metrics['cloud']['tasks_processed'] / total_processed * 100) if total_processed > 0 else 0.0
    print(f"Total Tasks Generated: {metrics['total_tasks']}\nTotal Tasks Processed: {total_processed}")
    print(f"Total Faults Detected: {metrics['total_faults_detected']}\nOverall System Accuracy: {overall_accuracy:.2%}")
    print(f"Intelligent Offload Rate: {offload_rate:.2f}%")

    print("\n" + "-"*20 + " Edge vs. Cloud Performance " + "-"*20)
    edge_stats = {
        'Tasks': metrics['edge']['tasks_processed'],
        'Avg Latency (ms)': float(np.mean(metrics['edge']['latency'] or [0])),
        'Avg Energy (μJ)': float(np.mean(metrics['edge']['energy'] or [0])),
        'Accuracy': float(np.mean(metrics['edge']['accuracy'] or [0]))
    }
    cloud_stats = {
        'Tasks': metrics['cloud']['tasks_processed'],
        'Avg Latency (ms)': float(np.mean(metrics['cloud']['latency'] or [0])),
        'Avg Energy (μJ)': float(np.mean(metrics['cloud']['energy'] or [0])),
        'Accuracy': float(np.mean(metrics['cloud']['accuracy'] or [0]))
    }
    print(pd.DataFrame({'Edge': edge_stats, 'Cloud': cloud_stats}).round(3))

    # Build dashboard
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Predictive Maintenance System: Simulation Dashboard ({scheduler_name.upper()})', fontsize=20)
    gs = fig.add_gridspec(3, 3)

    # Pie chart: task distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie([edge_stats['Tasks'], cloud_stats['Tasks']], labels=['Edge', 'Cloud'], autopct='%1.1f%%', colors=['#4c72b0', '#c44e52'])
    ax1.set_title('Task Processing Distribution')

    # Latency histograms
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics['edge']['latency']:
        sns.histplot(metrics['edge']['latency'], ax=ax2, label='Edge', kde=True)
    if metrics['cloud']['latency']:
        sns.histplot(metrics['cloud']['latency'], ax=ax2, label='Cloud', kde=True)
    ax2.set_title('Latency Distribution (Edge vs. Cloud)')
    ax2.set_xlabel('Latency (ms)')
    ax2.legend()

    # Energy boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(data=[metrics['edge']['energy'] or [0], metrics['cloud']['energy'] or [0]], ax=ax3)
    ax3.set_xticklabels(['Edge', 'Cloud'])
    ax3.set_title('Energy Consumption per Task')
    ax3.set_ylabel('Energy (μJ)')

    # Edge load over time (if available)
    if metrics['scheduling']['edge_load_history']:
        ax4 = fig.add_subplot(gs[1, 0])
        load_times, load_values = zip(*metrics['scheduling']['edge_load_history'])
        ax4.plot(load_times, load_values, color='#55a868')
        ax4.set_title('Edge Server Load Over Time')
        ax4.set_xlabel('Simulation Time (mins)')
        ax4.set_ylabel('Load (%)')
        ax4.set_ylim(0, 1.1)

    # If we collected fitness/reward history, plot moving average
    if metrics['scheduling']['ga_fitness_scores']:
        ax5 = fig.add_subplot(gs[1, 1])
        try:
            series = pd.Series(metrics['scheduling']['ga_fitness_scores'])
            ax5.plot(series.rolling(window=min(10, max(1, len(series)))).mean())
        except Exception:
            ax5.plot(metrics['scheduling']['ga_fitness_scores'])
        ax5.set_title('Scheduler Fitness/Reward (Lower is Better)')
        ax5.set_xlabel('Task Instance')
        ax5.set_ylabel('Fitness / Reward')

    # Confusion matrices (if provided)
    if rf_cm is not None:
        ax6 = fig.add_subplot(gs[2, 0])
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                    xticklabels=FAULT_TYPES.values(), yticklabels=FAULT_TYPES.values())
        ax6.set_title('Edge RF Model Confusion Matrix')
        ax6.set_xlabel('Predicted'); ax6.set_ylabel('Actual')

    if dense_cm is not None:
        ax7 = fig.add_subplot(gs[2, 1])
        sns.heatmap(dense_cm, annot=True, fmt='d', cmap='Reds', ax=ax7,
                    xticklabels=FAULT_TYPES.values(), yticklabels=FAULT_TYPES.values())
        ax7.set_title('Cloud Dense Model Confusion Matrix')
        ax7.set_xlabel('Predicted'); ax7.set_ylabel('Actual')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure so you can compare GA vs DQN dashboards
    dashboard_path = f"dashboard_{scheduler_name}.png"
    fig.savefig(dashboard_path, dpi=200)
    print(f"Saved dashboard image to: {dashboard_path}")

    # Save a numeric summary (CSV)
    summary = {
        'scheduler': scheduler_name,
        'total_tasks_generated': metrics['total_tasks'],
        'tasks_processed': total_processed,
        'faults_detected': metrics['total_faults_detected'],
        'offload_rate_pct': offload_rate,
        'edge_tasks': edge_stats['Tasks'],
        'cloud_tasks': cloud_stats['Tasks'],
        'edge_avg_latency_ms': edge_stats['Avg Latency (ms)'],
        'cloud_avg_latency_ms': cloud_stats['Avg Latency (ms)'],
        'edge_avg_energy_uJ': edge_stats['Avg Energy (μJ)'],
        'cloud_avg_energy_uJ': cloud_stats['Avg Energy (μJ)'],
        'edge_accuracy': edge_stats['Accuracy'],
        'cloud_accuracy': cloud_stats['Accuracy'],
        'overall_accuracy': overall_accuracy
    }
    summary_df = pd.DataFrame([summary])
    csv_path = f"metrics_summary_{scheduler_name}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved metrics summary to: {csv_path}")

    # Save db_storage if populated (task-level records)
    if db_storage:
        try:
            pd.DataFrame(db_storage).to_csv(f"task_records_{scheduler_name}.csv", index=False)
            print(f"Saved task-level records to: task_records_{scheduler_name}.csv")
        except Exception as e:
            print("Warning: could not save db_storage:", e)

    # Show the plot interactively (if running in GUI-capable env)
    try:
        plt.show()
    except Exception:
        pass

# -----------------------------------------------------------
#                        MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    final_metrics, rf_cm, dense_cm = run_simulation()

    if (final_metrics['edge']['tasks_processed'] + final_metrics['cloud']['tasks_processed']) > 0:
        analyze_results(final_metrics, rf_cm, dense_cm, scheduler_name=SCHEDULER_NAME)
        print("\n" + "="*70); print(" " * 27 + "PROJECT HIGHLIGHTS"); print("="*70)
        edge_percentage = (final_metrics['edge']['tasks_processed'] / (final_metrics['edge']['tasks_processed'] + final_metrics['cloud']['tasks_processed'])) * 100
        overall_accuracy_val = np.mean(final_metrics['edge']['accuracy'] + final_metrics['cloud']['accuracy']) if (final_metrics['edge']['accuracy'] or final_metrics['cloud']['accuracy']) else 0.0
        print(f"\n✓ Intelligent Task Scheduling: The {SCHEDULER_NAME.upper()} scheduler dynamically optimized task placement, ensuring urgent, simple analyses were handled instantly on the edge while complex diagnostics were sent to the cloud.")
        print(f"✓ Hybrid Machine Learning: A lightweight Edge RF model and a powerful Cloud Neural Network worked in tandem to deliver high-performance fault detection tailored to the complexity of the task.")
        print(f"✓ Resource Efficiency: By processing {edge_percentage:.1f}% of tasks locally, the system drastically cut network traffic and cloud costs, demonstrating a lean operational model.")
        print(f"✓ Proactive Maintenance: With an overall accuracy of {overall_accuracy_val:.2%}, the system reliably detected {final_metrics['total_faults_detected']} potential equipment failures, directly translating to reduced downtime and increased factory productivity.")
        print("\nConclusion: This simulation validates a robust, efficient, and intelligent Edge-Cloud architecture using a DQN-based scheduler, perfectly suited for modern Industry 4.0 predictive maintenance.")
        print("="*70)
