import numpy as np
import torch


class PrioritizedReplayStorage:
    def __init__(
        self, buffer_size, alpha=0.6, beta=0.4, beta_schedule=None, epsilon=1e-6
    ):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.beta_schedule = beta_schedule
        self.buffer = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        if self.size < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        if self.size == self.buffer_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.size]
        prob = priorities**self.alpha
        prob /= prob.sum()
        indices = np.random.choice(self.size, batch_size, p=prob)
        samples = [self.buffer[i] for i in indices]
        weights = (self.size * prob[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        if self.beta_schedule is not None:
            self.beta = min(self.beta_schedule, self.beta + self.beta_schedule)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority + self.epsilon
