import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import ast

# Step 1: Read and parse the CSV file
array_columns = [
    'd_S_real', 'd_S_imag', 'y_MT_ZF_real', 'y_MT_ZF_imag',
    'g_MT_real', 'g_MT_imag', 'theta', 'phi_MT', 'phi_S',
    'alpha_S_real', 'alpha_S_imag'
]

converters = {col: ast.literal_eval for col in array_columns}

df = pd.read_csv('mmwave_data.csv', converters=converters)

# Step 2: Define the environment
class BeamSelectionEnv:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.n_samples = len(df)
        self.current_index = 0

        # Define action space (e.g., number of possible beams)
        self.action_space = 8  # Adjust based on your system's possible actions

        # Define state space size
        sample_state = self.get_state(0)
        self.state_size = len(sample_state)

    def reset(self):
        self.current_index = 0
        state = self.get_state(self.current_index)
        return state

    def get_state(self, index):
        row = self.df.iloc[index]

        # Flatten array features into a single vector
        state_features = []

        for col in self.df.columns:
            if isinstance(row[col], list):
                # Flatten nested lists
                flat_list = list(self.flatten(row[col]))
                state_features.extend(flat_list)
            else:
                # Include scalar values directly
                state_features.append(row[col])

        state = np.array(state_features, dtype=np.float32)
        return state

    def flatten(self, l):
        # Recursively flatten nested lists
        for el in l:
            if isinstance(el, list):
                yield from self.flatten(el)
            else:
                yield el

    def step(self, action):
        # Get the current row
        row = self.df.iloc[self.current_index]

        # Compute the reward based on action
        reward = self.compute_reward(row, action)

        # Check if done
        done = self.current_index >= self.n_samples - 1

        # Move to next index
        self.current_index += 1
        if not done:
            next_state = self.get_state(self.current_index)
        else:
            next_state = None

        return next_state, reward, done, {}

    def compute_reward(self, row, action):
        # Placeholder: Replace with actual computation based on your system
        # For example, calculate achievable rate or SNR based on action
        reward = np.random.random()  # Random reward as a placeholder
        return reward

# Step 3: Define the agent
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.out(x)
        return x

# Step 4: Training the agent
def train_agent(env, agent, episodes=100, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    memory = []
    batch_size = 64  # Adjusted for larger dataset

    for e in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)

        total_reward = 0
        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() <= epsilon:
                action = random.randrange(env.action_space)
            else:
                with torch.no_grad():
                    q_values = agent(state)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if next_state is not None:
                next_state = torch.FloatTensor(next_state)

            # Store experience in memory
            memory.append((state, action, reward, next_state, done))

            # Limit memory size to 10000
            if len(memory) > 10000:
                memory.pop(0)

            # If memory is large enough, start training
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                states_mb = torch.stack([mb[0] for mb in minibatch])
                actions_mb = torch.tensor([mb[1] for mb in minibatch], dtype=torch.int64)
                rewards_mb = torch.tensor([mb[2] for mb in minibatch], dtype=torch.float32)
                next_states_mb = torch.stack([mb[3] for mb in minibatch if mb[3] is not None])
                dones_mb = torch.tensor([mb[4] for mb in minibatch], dtype=torch.bool)

                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = agent(next_states_mb)
                    max_next_q_values = torch.max(next_q_values, dim=1)[0]
                    target_q_values = rewards_mb
                    target_q_values[~dones_mb] += gamma * max_next_q_values

                # Compute current Q-values
                current_q_values = agent(states_mb).gather(1, actions_mb.unsqueeze(1)).squeeze()

                # Compute loss
                loss = criterion(current_q_values, target_q_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        # Decrease epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# Step 5: Main function
if __name__ == "__main__":
    env = BeamSelectionEnv(df)
    state_size = env.state_size
    action_size = env.action_space

    agent = DQNAgent(state_size, action_size)

    train_agent(env, agent, episodes=100)
