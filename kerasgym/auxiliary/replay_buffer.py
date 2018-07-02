import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Stores agent's experience, to be used for experience replay learning.

    Always stores the following: states, actions, rewards, next states, done flags.

    Parameters:
        buffer_size (int): limit for how many observations to hold in total.
        state_dims (tuple(int)): shape, not including observation dim, of environment state.
        action_space (gym.spaces.*): whole action space object from environment.
        state_dtype (type): numpy datatype for states to be stored in.
                            default: np.float32.

    Member variables:
        buffer_size (int): limit for how many observations are held in total.
        current_idx (int): current position in `buffer`; circular.
        current_size (int): number of items in `buffer`; caps at `buffer_size`.
        action_space (gym.spaces.*): whole action space object from environment.
        action_type (str): class name of action space.
        state_dtype (type): numpy datatype for states to be stored in.
        buffer (dict): all observations as dictionary of arrays.
    """
    def __init__(self, buffer_size, state_dims, action_space, state_dtype=np.float32):
        self.buffer_size = buffer_size
        self.current_idx = 0
        self.current_size = 0
        self.action_space = action_space
        self.action_type = action_space.__class__.__name__.lower()
        self.state_dtype = state_dtype
        self.buffer = self._initialize_buffer(state_dims)

    def _initialize_buffer(self, state_dims):
        """Init helper to create self.buffer.

        Parameters:
            state_dims (tuple(int)): shape, not including observation dim, of environment state.
        """
        buffer = {}
        buffer['states'] = np.zeros([self.buffer_size] + state_dims, dtype=self.state_dtype)
        buffer['next_states'] = np.zeros([self.buffer_size] + state_dims, dtype=self.state_dtype)
        buffer['rewards'] = np.zeros(self.buffer_size)
        buffer['dones'] = np.zeros(self.buffer_size)
        if self.action_type == 'discrete':
            buffer['actions'] = np.zeros((self.buffer_size, self.action_space.n))
        elif self.action_type == 'multidiscrete':
            buffer['actions'] = [np.zeros((self.buffer_size, n)) for n in self.action_space.nvec]
        elif self.action_type == 'box':
            buffer['actions'] = np.zeros((self.buffer_size, self.action_space.shape))
        elif self.action_type == 'multibinary':
            buffer['actions'] = [np.zeros((self.buffer_size, 2)) for _ in range(self.action_space.n)]
        return buffer

    def get_batch(self, batch_size):
        """Return random sample of observations from buffer for all keys, as dictionary.

        Parameters:
            batch_size (int): number of observations to return.
        """
        inds = np.random.randint(self.current_size, size=batch_size)
        if self.action_type in ['discrete', 'box']:
            actions = self.buffer['actions'][inds,:]
        elif self.action_type in ['multidiscrete', 'multibinary']:
            actions = [a[inds,:] for a in self.buffer['actions']]
        return {'states': self.buffer['states'][inds,:],
                'actions': actions,
                'rewards': self.buffer['rewards'][inds],
                'next_states': self.buffer['next_states'][inds,:],
                'dones': self.buffer['dones'][inds]}

    def add(self, state, action, reward, next_state, done):
        """Add single observation to all keys of buffer.

        Parameters:
            state, action, reward, next_state, done: observation to be added.
        """
        self.buffer['states'][self.current_idx,:] = state
        if self.action_type in ['discrete', 'box']:
            self.buffer['actions'][self.current_idx,:] = action
        elif self.action_type in ['multidiscrete', 'multibinary']:
            for i in range(len(self.actions)):
                self.buffer['actions'][i][self.current_idx,:] = action[i]
        self.buffer['rewards'][self.current_idx] = reward
        self.buffer['next_states'][self.current_idx,:] = next_state
        self.buffer['dones'][self.current_idx] = done

        if self.current_size < self.buffer_size:
            self.current_size += 1
        self.current_idx = (self.current_idx + 1) % self.buffer_size
