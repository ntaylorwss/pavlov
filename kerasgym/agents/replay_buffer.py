import random
import numpy as np
from collections import deque
from ..util import get_action_type


class ReplayBuffer:
    def __init__(self, buffer_size, state_dims, action_space, state_dtype):
        self.current_idx = 0
        self.current_size = 0
        self.buffer_size = buffer_size
        self.action_space = action_space
        self.action_type = get_action_type(self.action_space)
        self.state_dtype = state_dtype
        self._initialize_buffer(state_dims)

    def _initialize_buffer(self, state_dims):
        self.states = np.zeros([self.buffer_size] + state_dims, dtype=self.state_dtype)
        self.next_states = np.zeros([self.buffer_size] + state_dims, dtype=self.state_dtype)
        self.rewards = np.zeros(self.buffer_size)
        self.dones = np.zeros(self.buffer_size)
        if self.action_type == 'discrete':
            self.actions = np.zeros((self.buffer_size, self.action_space.n))
        elif self.action_type == 'multidiscrete':
            self.actions = [np.zeros((self.buffer_size, n)) for n in self.action_space.nvec]
        elif self.action_type == 'box':
            self.actions = np.zeros((self.buffer_size, self.action_space.shape))
        elif self.action_type == 'multibinary':
            self.actions = [np.zeros((self.buffer_size, 2)) for _ in range(self.action_space.n)]

    def get_batch(self, batch_size):
        inds = np.random.randint(self.current_size, size=batch_size)
        if self.action_type in ['discrete', 'box']:
            actions = self.actions[inds,:]
        elif self.action_type in ['multidiscrete', 'multibinary']:
            actions = [a[inds,:] for a in self.actions]
        return {'states': self.states[inds,:],
                'actions': actions,
                'rewards': self.rewards[inds],
                'next_states': self.next_states[inds,:],
                'dones': self.dones[inds]}

    def add(self, state, action, reward, next_state, done):
        self.states[self.current_idx,:] = state
        if self.action_type in ['discrete', 'box']:
            self.actions[self.current_idx,:] = action
        elif self.action_type in ['multidiscrete', 'multibinary']:
            for i in range(len(self.actions)):
                self.actions[i][self.current_idx,:] = action[i]
        self.rewards[self.current_idx] = reward
        self.next_states[self.current_idx,:] = next_state
        self.dones[self.current_idx] = done

        if self.current_size < self.buffer_size:
            self.current_size += 1
        self.current_idx = (self.current_idx + 1) % self.buffer_size
