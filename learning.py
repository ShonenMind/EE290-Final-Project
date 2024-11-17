# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Dataset
data = {
    'SBS_ID': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'MT_Location_X': [120.5, 85.3, 95.7, 150.2, 165.8, 178.3, 90.5, 110.2, 130.8],
    'MT_Location_Y': [45.2, 65.8, 78.9, 110.5, 95.2, 88.7, 150.3, 165.8, 145.2],
    'SINR_dB': [15.8, 12.3, 11.5, 18.2, 16.5, 14.8, 13.2, 11.8, 10.5],
    'Path_Loss_dB': [85.3, 90.2, 92.5, 82.1, 84.5, 86.8, 88.9, 91.2, 93.5],
    'Angular_Position_Degrees': [45, 30, 60, 120, 150, 135, 90, 75, 60],
    'Channel_Quality_Indicator': [8, 6, 5, 9, 8, 7, 6, 5, 4],
    'Interference_Level_dBm': [-95.2, -92.5, -90.8, -98.5, -96.2, -94.5, -93.8, -91.5, -89.2],
    'Selected_Beam_Index': [3, 2, 4, 1, 2, 3, 4, 3, 2],  # Label/Target variable
    'Number_Active_MTs': [5, 5, 5, 4, 4, 4, 6, 6, 6],
    'Distance_To_MT_Meters': [25.3, 35.8, 42.1, 20.5, 28.7, 35.2, 45.8, 52.3, 58.9]
}

# DataFrame Converting
df = pd.DataFrame(data)

# Environment
class BeamSelectionEnv:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.n_samples = len(df)
        self.action_space = df['Selected_Beam_Index'].nunique()
        self.reset()

    def reset(self):
        self.current_index = 0
        state = self.get_state(self.current_index)
        return state

    def get_state(self, index):
        # Exclude the 'Selected_Beam_Index' from the state
        state = self.df.iloc[index].drop('Selected_Beam_Index').values.astype(np.float32)
        return state

    def step(self, action):
        # Get the correct action
        correct_action = self.df.iloc[self.current_index]['Selected_Beam_Index'] - 1  # Adjust for zero-indexing

        # Define reward
        if action == correct_action:
            reward = 1.0
        else:
            reward = -1.0  # Penalize wrong actions

        # Check if done
        done = self.current_index >= self.n_samples - 1

        # Move to next index
        self.current_index += 1
        if not done:
            next_state = self.get_state(self.current_index)
        else:
            next_state = None

        return next_state, reward, done, {}

# Define the DQN Agent
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
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

# Training the agent
def train_agent(env, agent, episodes=1000, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    memory = []
    batch_size = 16

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

            # If memory is large enough, start training
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, s_next, d in minibatch:
                    target = r
                    if not d:
                        with torch.no_grad():
                            target = r + gamma * torch.max(agent(s_next)).item()
                    current_q = agent(s)[a]
                    loss = criterion(current_q, torch.tensor(target))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            state = next_state

        # Decrease epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (e+1) % 100 == 0:
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

# Main code
if __name__ == "__main__":
    env = BeamSelectionEnv(df)

    state_size = len(env.get_state(0))
    action_size = env.action_space  # Number of unique beam indices

    agent = DQNAgent(state_size, action_size)

    train_agent(env, agent, episodes=1000)
