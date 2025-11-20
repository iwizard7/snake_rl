import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor (policy) head
        self.actor_head = nn.Linear(hidden_size, output_size)
        # Critic (value) head
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        policy_logits = self.actor_head(x)
        value = self.critic_head(x)
        
        return policy_logits, value

    def save(self, file_name='a2c_model.pth'):
        torch.save(self.state_dict(), file_name)

class A2CAgent:
    def __init__(self, input_size=10, hidden_size=64, output_size=3, lr=0.001, gamma=0.99, entropy_coef=0.01, value_coef=0.5):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        self.model = ActorCriticNet(input_size, hidden_size, output_size).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).to(DEVICE).unsqueeze(0)
        policy_logits, _ = self.model(state_tensor)
        
        # Sample from policy
        action_probs = torch.softmax(policy_logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        
        return action, dist.log_prob(torch.tensor(action).to(DEVICE))

    def compute_returns(self, rewards, dones, last_value=None):
        returns = []
        R = last_value if last_value is not None else 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float).to(DEVICE)
        return returns

    def train_step(self, states, actions, log_probs, returns, values):
        states = torch.tensor(np.array(states), dtype=torch.float).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        log_probs = torch.stack(log_probs)
        
        # Get current policy and values
        policy_logits, pred_values = self.model(states)
        
        # Advantage function
        advantages = returns - pred_values.squeeze()
        
        # Actor loss (policy gradient)
        action_probs = torch.softmax(policy_logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(new_log_probs - log_probs)
        actor_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(pred_values.squeeze(), returns)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
