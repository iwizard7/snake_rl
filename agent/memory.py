import random
from collections import deque
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self, max_size, prioritized=False):
        self.buffer = deque(maxlen=max_size)
        self.prioritized = prioritized
        if self.prioritized:
            self.priorities = deque(maxlen=max_size)
            self.alpha = 0.6  # Priority exponent
            self.beta = 0.4   # Importance sampling exponent
            self.beta_increment = 0.001
            self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if self.prioritized:
            self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        if self.prioritized:
            # Priority sampling
            priorities = np.array(self.priorities, dtype=np.float32)
            probs = priorities ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            batch = [self.buffer[i] for i in indices]

            # Importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)

            states, actions, rewards, next_states, dones = zip(*batch)

            sample_info = (indices, weights)
        else:
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            sample_info = None

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
            sample_info
        )

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-5) ** self.alpha
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)

    def update_beta(self):
        if self.prioritized:
            self.beta = min(1.0, self.beta + self.beta_increment)

    def __len__(self):
        return len(self.buffer)
