import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agent.memory import Memory
from agent.state_builder import StateBuilder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(action, dtype=torch.long).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float).to(DEVICE)
        done = torch.tensor(done, dtype=torch.bool).to(DEVICE)

        if len(state.shape) == 1:
            # Single sample
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Current Q values
        pred = self.model(state)
        Q_new = reward + self.gamma * torch.max(self.model(next_state), dim=1)[0]

        # Q_new = reward if done, else reward + gamma * max Q
        Q_new = Q_new * (~done) + reward * done

        # Target Q
        target = pred.clone()
        for idx in range(len(action)):
            target[idx][action[idx]] = Q_new[idx]

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss.item()

class DQNAgent:
    def __init__(self, input_size=10, hidden_size=64, output_size=3, lr=0.001, gamma=0.9, epsilon=1.0, eps_decay=0.995):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.memory_size = 100000
        self.batch_size = 1000
        
        self.memory = Memory(self.memory_size)
        self.state_builder = StateBuilder()
        
        self.model = Linear_QNet(input_size, hidden_size, output_size).to(DEVICE)
        self.trainer = QTrainer(self.model, lr, gamma)

    def get_state(self, game):
        return self.state_builder.build_state(game)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            # Random action
            action = np.random.randint(0, self.output_size)
        else:
            # Best action
            state_tensor = torch.tensor(state, dtype=torch.float).to(DEVICE).unsqueeze(0)
            pred = self.model(state_tensor)
            action = torch.argmax(pred).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = mini_sample
            loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
            return loss
        return None
        
    def decay_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon *= self.eps_decay
