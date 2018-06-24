import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size, state_dims, action_space, state_dtype):
        self.current_idx = 0
        self.current_size = 0
        self.buffer_size = buffer_size
        self.action_space = action_space
        self.dtype = dtype
        self.keys = keys
        self.buffer = self._initialize_buffer(state_dims)

    def _initialize_buffer(self, state_dims):
        self.states = np.zeros([buffer_size] + state_dims, dtype=state_dtype)
        self.next_states = np.zeros([buffer_size] + state_dims, dtype=state_dtype)
        self.rewards = np.zeros(buffer_size)
        self.dones = np.zeros(buffer_size)
        # TODO: self.actions


    def get_batch(self, batch_size):
        inds = np.random.randint(self.current_size, size=batch_size)
        actions = # TODO
        return {'states': self.states[inds,:],
                'actions': actions,
                'rewards': self.rewards[inds],
                'next_states': self.next_states[inds,:],
                'dones': self.dones[inds]}

    def add(self, state, action, reward, next_state, done):
        self.states[self.current_idx,:] = state
        # TODO: self.actions
        self.rewards[self.current_idx] = reward
        self.next_states[self.current_idx,:] = next_state
        self.dones[self.current_idx] = done

        if self.current_size < self.buffer_size:
            self.current_size += 1
        self.current_idx = (self.current_idx + 1) % self.buffer_size
