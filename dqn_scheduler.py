# dqn_scheduler.py
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ----------------------------
# Environment (simple RL env)
# ----------------------------
class TaskEnv:
    """
    Simple environment: each step produces a new task with features:
    - task_complexity (0.5 .. 3.0)
    - edge_load (0.0 .. 1.0)
    - cloud_queue (0.0 .. 5.0)
    - net_latency (1 .. 10)  (ms)
    Action: 0 -> Edge, 1 -> Cloud
    Reward: negative of GA-style fitness (so maximizing reward = minimizing cost)
    """
    def __init__(self):
        self.step_count = 0

    def sample_task(self):
        # sample realistic task and system parameters
        task_complexity = np.random.uniform(0.5, 3.0)   # low..high complexity
        edge_load = np.random.beta(2, 5)  # skewed towards low load, range 0..1
        cloud_queue = np.random.exponential(1.0)  # avg ~1.0
        net_latency = np.random.uniform(2.0, 8.0)
        return np.array([task_complexity, edge_load, cloud_queue, net_latency], dtype=np.float32)

    def reset(self):
        self.step_count = 0
        return self.sample_task()

    def fitness(self, action, state):
        # replicate your GA fitness logic (slightly adapted, deterministic)
        task_complexity, edge_load, cloud_queue, net_latency = state
        if action == 0:  # Edge
            latency = (task_complexity * 2.0) + (edge_load * 10.0 if edge_load > 0.8 else edge_load * 3.0)
            energy = task_complexity * 0.5
            qos_penalty = 20.0 if edge_load > 0.9 else 0.0
        else:  # Cloud
            latency = net_latency + (task_complexity * 1.5) + (cloud_queue * 0.5)
            energy = (task_complexity * 0.2) + 0.3
            qos_penalty = 0.0
        fitness_val = 0.4 * latency + 0.3 * energy + 0.3 * qos_penalty
        return fitness_val, latency, energy, qos_penalty

    def step(self, action):
        fitness_val, latency, energy, qos = self.fitness(action, self.current_state)
        # reward is negative cost (agent wants to maximize)
        reward = -fitness_val
        done = False  # episodic control is in training loop
        info = {'latency': latency, 'energy': energy, 'qos': qos, 'fitness': fitness_val}
        # next state: sample new task (non-Markovian in strict sense but OK here)
        self.current_state = self.sample_task()
        self.step_count += 1
        return self.current_state, reward, done, info

# ----------------------------
# DQN Agent
# ----------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 lr=1e-3, gamma=0.99, batch_size=64,
                 mem_size=50000, target_update_freq=50):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=mem_size)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq

        # main and target networks
        self.model = self._build_model(lr)
        self.target_model = self._build_model(lr)
        self.update_target_network()

    def _build_model(self, lr):
        model = Sequential([
            InputLayer(input_shape=(self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, greedy=False):
        if (not greedy) and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return int(np.argmax(q))

    def replay(self):
        if len(self.memory) < max(self.batch_size, 100):
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        # predict Q(s,a) and target Q
        q_values = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)
        q_target = q_values.copy()

        for i in range(self.batch_size):
            if dones[i]:
                q_target[i, actions[i]] = rewards[i]
            else:
                q_target[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])

        history = self.model.fit(states, q_target, epochs=1, verbose=0, batch_size=self.batch_size)
        loss = history.history['loss'][0]
        self.learn_step_counter += 1

        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

# ----------------------------
# Training loop
# ----------------------------
def train_dqn(episodes=500, steps_per_episode=50):
    env = TaskEnv()
    state_dim = 4
    action_dim = 2
    agent = DQNAgent(state_dim, action_dim,
                     lr=1e-3, gamma=0.98,
                     batch_size=64, mem_size=20000, target_update_freq=100)

    # warm-up: fill some memory with random interactions
    print("Warming up memory with random actions...")
    env.current_state = env.sample_task()
    for _ in range(2000):
        s = env.current_state
        a = np.random.randint(0, action_dim)
        ns, r, _, _ = env.step(a)
        agent.remember(s, a, r, ns, False)

    print("Starting training...")
    reward_history = []
    loss_history = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        env.current_state = state
        ep_reward = 0.0
        ep_loss = 0.0
        for step in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss is not None:
                ep_loss += loss
            state = next_state
            ep_reward += reward

        reward_history.append(ep_reward)
        loss_history.append(ep_loss / (steps_per_episode if ep_loss else 1.0))

        if ep % 20 == 0 or ep == 1:
            avg_r = np.mean(reward_history[-20:])
            avg_loss = np.mean(loss_history[-20:])
            print(f"Episode {ep:3d} | AvgReward(20): {avg_r: .3f} | Epsilon: {agent.epsilon: .3f} | AvgLoss: {avg_loss: .4f}")

    print("Training completed.")
    return agent, reward_history, loss_history

# ----------------------------
# Quick evaluation routine
# ----------------------------
def evaluate_agent(agent, episodes=100):
    env = TaskEnv()
    total_rewards = []
    stats = {'edge_count':0, 'cloud_count':0}
    for ep in range(episodes):
        state = env.reset()
        env.current_state = state
        ep_reward = 0.0
        for _ in range(20):
            action = agent.act(state, greedy=True)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            # track action counts
            if action == 0: stats['edge_count'] += 1
            else: stats['cloud_count'] += 1
            state = next_state
        total_rewards.append(ep_reward)
    avg_reward = np.mean(total_rewards)
    print(f"Evaluation over {episodes} episodes: AvgReward={avg_reward:.3f}, EdgeActions={stats['edge_count']}, CloudActions={stats['cloud_count']}")
    return avg_reward, stats

# ----------------------------
# Run training + evaluation
# ----------------------------
if __name__ == "__main__":
    agent, rewards, losses = train_dqn(episodes=300, steps_per_episode=40)
    print("\nEvaluating trained agent (greedy)...")
    evaluate_agent(agent, episodes=200)
    # Save model
    agent.model.save("dqn_scheduler_model.h5")
    print("Model saved to dqn_scheduler_model.h5")
m