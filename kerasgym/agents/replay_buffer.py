import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = {k: deque(maxlen=buffer_size)
                       for k in ['states', 'actions', 'rewards', 'next_states', 'dones']}
        self.current_size = 0

    def get_batch(self, batch_size):
        inds = random.sample(range(self.current_size), batch_size)
        return {k: np.array(self.buffer[k])[inds] for k in self.buffer}

    def add(self, state, action, reward, next_state, done):
        self.buffer['states'].append(state)
        self.buffer['next_states'].append(next_state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        if self.current_size < self.buffer_size:
            self.current_size += 1

    def erase(self):
        self.__init__(self.buffer_size)
