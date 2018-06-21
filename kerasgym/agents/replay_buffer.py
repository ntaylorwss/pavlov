import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size, state_dims, action_dim, state_dtype,
                 keys=['states','actions','rewards','next_states','dones']):
        self.current_idx = 0
        self.current_size = 0
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.keys = keys
        self.buffer = self._initialize_buffer(state_dims, action_dim)

    def _initialize_buffer(self, state_dims, action_dim):
        buf = {}
        for key in self.keys:
            if key in ['states', 'next_states']:
                buf[key] = np.zeros([buffer_size] + state_dims, dtype=state_dtype)
            elif key == 'actions':
                buf[key] = np.zeros((buffer_size, action_dim))
            else:
                buf[key] = np.zeros(buffer_size)

    def get_batch(self, batch_size):
        inds = np.random.randint(self.current_size, size=batch_size)
        out = {}
        for key in self.buffer:
            if len(self.buffer[key].shape) > 1:
                out[key] = self.buffer[key][inds,:]
            else:
                out[key] = self.buffer[key][inds]
        return out

    def add(self, **keyed_obs):
        for key, value in keyed_obs.items():
            if key in ['states', 'next_states']:
                self.buffer[key][self.current_idx] = value.astype(self.state_dtype)
            else:
                self.buffer[key][self.current_idx] = value

        if self.current_size < self.buffer_size:
            self.current_size += 1
        self.current_idx = (self.current_idx + 1) % self.buffer_size
