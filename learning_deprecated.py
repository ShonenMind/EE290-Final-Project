import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import ast
import matplotlib.pyplot as plt

# Step 1: Read and Parse the CSV file
array_columns = [
    'd_S_real', 'd_S_imag',
    'H_S_norms', 'H_S_singular_values', 'H_S_eigenvalues',
    'g_MT_real', 'g_MT_imag',
    'theta', 'phi_MT', 'phi_S',
    'alpha_S_real', 'alpha_S_imag',
    'optimal_beams',
    'sinr_values'
]

def parse_array(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

df = pd.read_csv('mmwave_data.csv', converters={col: parse_array for col in array_columns})

# Step 2: Define the environment
class BeamSelectionEnv:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.n_samples = len(df)
        self.current_index = 0

        # Define the action space
        beam_indices = df['optimal_beams'].apply(lambda x: max(x) if x else 0)
        self.action_space = max(beam_indices) + 1  # Beam indices start from 0

        # Define state size
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
            if col not in ['optimal_beams', 'sinr_values', 'asr']:  # Exclude target and reward columns
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

        # Get optimal beams for this sample
        optimal_beams = row['optimal_beams']

        # Compute the reward
        # Option 1: Use asr as reward
        reward = row['asr']

        # Option 2: Use sinr_values corresponding to the action
        # sinr_values = row['sinr_values']
        # reward = sinr_values[action] if action < len(sinr_values) else 0.0

        # Check if done
        done = self.current_index >= self.n_samples - 1

        # Move to next index
        self.current_index += 1
        if not done:
            next_state = self.get_state(self.current_index)
        else:
            next_state = None

        return next_state, reward, done, {}

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
def train_agent(env, agent, episodes=10, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    memory = []
    batch_size = 64  # Adjusted for larger dataset

    losses = []     # To store loss per episode
    rewards = []    # To store total reward per episode
    accuracies = [] # To store accuracy per episode

    for e in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)

        total_reward = 0
        total_loss = 0
        total_correct = 0
        total_steps = 0
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
            
            # Update accuracy metrics
            total_steps += 1
            if reward == 1.0:
                total_correct += 1

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
                    target_q_values = rewards_mb.clone()
                    target_q_values[~dones_mb] += gamma * max_next_q_values

                # Compute current Q-values
                current_q_values = agent(states_mb).gather(1, actions_mb.unsqueeze(1)).squeeze()

                # Compute loss
                loss = criterion(current_q_values, target_q_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            state = next_state
            total_steps += 1

        # Decrease epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Calculate average loss and accuracy
        avg_loss = total_loss / (total_steps if total_steps > 0 else 1)
        accuracy = total_correct / (total_steps if total_steps > 0 else 1)

        losses.append(avg_loss)
        rewards.append(total_reward)
        accuracies.append(accuracy)

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Epsilon: {epsilon:.2f}")

    return losses, rewards, accuracies

# Step 5: Main function
if __name__ == "__main__":
    env = BeamSelectionEnv(df)
    state_size = env.state_size
    action_size = env.action_space

    agent = DQNAgent(state_size, action_size)

    losses, rewards, accuracies = train_agent(env, agent, episodes=10)

    # Plot loss and total reward over episodes
    episodes_range = range(1, len(losses)+1)
    plt.figure(figsize=(12,5))

    # Plot Loss
    plt.subplot(1,2,1)
    plt.plot(episodes_range, losses, label='Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Episodes')
    plt.legend()

    # Plot Total Reward
    plt.subplot(1,2,2)
    plt.plot(episodes_range, rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1,2,3)
    plt.plot(episodes_range, accuracies, label='Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.show()

