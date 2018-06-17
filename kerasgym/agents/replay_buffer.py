import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size, state_dims, action_dim, dtype):
        self.current_idx = 0
        self.current_size = 0
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.state_buffer = np.zeros([buffer_size] + state_dims, dtype=dtype)
        self.nextstate_buffer = np.zeros([buffer_size] + state_dims, dtype=dtype)
        self.action_buffer = np.zeros((buffer_size, action_dim))
        self.reward_buffer = np.zeros(buffer_size)
        self.done_buffer = np.zeros(buffer_size)

    def get_batch(self, batch_size):
        inds = np.random.randint(self.current_size, size=batch_size)
        return {'states': self.state_buffer[inds,:],
                'actions': self.action_buffer[inds,:],
                'rewards': self.reward_buffer[inds],
                'next_states': self.nextstate_buffer[inds,:],
                'dones': self.done_buffer[inds]}

    def add(self, state, action, reward, next_state, done):
        self.state_buffer[self.current_idx] = state.astype(self.dtype)
        self.nextstate_buffer[self.current_idx] = next_state.astype(self.dtype)
        self.action_buffer[self.current_idx] = action
        self.reward_buffer[self.current_idx] = reward
        self.done_buffer[self.current_idx] = done

        if self.current_size < self.buffer_size:
            self.current_size += 1
        self.current_idx = (self.current_idx + 1) % self.buffer_size
