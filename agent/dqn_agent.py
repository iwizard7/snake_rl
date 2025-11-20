import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agent.memory import Memory
from agent.state_builder import StateBuilder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.hidden_size = hidden_size
        
        # Common layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        if self.dueling:
            # Dueling: Value stream
            self.value_net = nn.Linear(hidden_size, 1)
            # Advantage stream
            self.advantage_net = nn.Linear(hidden_size, output_size)
        else:
            self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        if self.dueling:
            # Dueling architecture
            v = self.value_net(x)  # State value
            a = self.advantage_net(x)  # Action advantages
            
            # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        else:
            return self.linear3(x)

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma, use_double_dqn=True, prioritized_weights=None):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = Linear_QNet(model.linear1.in_features, model.hidden_size, model.linear3.out_features, dueling=model.dueling).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.use_double_dqn = use_double_dqn
        self.tau = 0.005  # Soft update rate
        self.update_count = 0

    def soft_update_target(self):
        """Soft update target network parameters"""
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update_target(self):
        """Hard update target network (every C steps)"""
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done, sample_info=None):
        state = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(action, dtype=torch.long).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float).to(DEVICE)
        done = torch.tensor(done, dtype=torch.bool).to(DEVICE)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Get importance weights for prioritized replay
        weights = torch.ones_like(reward, device=DEVICE)
        if sample_info is not None:
            indices, w = sample_info
            weights = torch.tensor(w, dtype=torch.float).to(DEVICE)

        # Current Q values
        pred = self.model(state)

        if self.use_double_dqn:
            # Double DQN: use main network to select action, target to evaluate
            next_actions = self.model(next_state).argmax(dim=1)
            Q_next = self.target_model(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            # Standard DQN
            Q_next = torch.max(self.target_model(next_state), dim=1)[0]

        Q_new = reward + self.gamma * Q_next * (~done)

        # Target Q
        target = pred.clone()
        for idx in range(len(action)):
            target[idx][action[idx]] = Q_new[idx]

        self.optimizer.zero_grad()
        
        # Weighted loss for prioritized replay
        loss = weights.unsqueeze(1) * self.criterion(target, pred)
        loss = loss.mean()
        
        loss.backward()
        self.optimizer.step()

        # Calculate TD errors for prioritized replay
        if sample_info is not None:
            indices, _ = sample_info
            td_errors = (target - pred).detach().cpu().numpy()
            td_errors = td_errors[range(len(action)), action.cpu().numpy()]
            return loss.item(), td_errors
        else:
            return loss.item(), None

class DQNAgent:
    def __init__(self, input_size=10, hidden_size=64, output_size=3, lr=0.001, gamma=0.9, epsilon=1.0, eps_decay=0.995,
                 use_double_dqn=True, use_dueling=False, use_prioritized_replay=False):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.memory_size = 100000
        self.batch_size = 1000
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_prioritized_replay = use_prioritized_replay
        
        self.memory = Memory(self.memory_size, prioritized=self.use_prioritized_replay)
        self.state_builder = StateBuilder()
        
        self.model = Linear_QNet(input_size, hidden_size, output_size, dueling=self.use_dueling).to(DEVICE)
        self.trainer = QTrainer(self.model, lr, gamma, use_double_dqn=self.use_double_dqn)

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
            states, actions, rewards, next_states, dones, sample_info = mini_sample
            loss, td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, sample_info)
            
            # Update priorities if prioritized replay
            if self.use_prioritized_replay and td_errors is not None:
                indices, _ = sample_info
                self.memory.update_priorities(indices, td_errors)
                self.memory.update_beta()  # Increase beta
            
            # Soft update target network
            self.trainer.soft_update_target()
            
            return loss
        return None
        
    def decay_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon *= self.eps_decay
