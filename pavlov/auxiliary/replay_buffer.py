import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Stores agent's experience, to be used for experience replay learning.

    Always stores the following: states, actions, rewards, next states, done flags.

    Parameters
    ----------
    buffer_size : int
        limit for how many observations to hold in total.
    state_dtype : type
        numpy datatype for states to be stored in (default is np.float32).

    Attributes
    ----------
    buffer_size : int
        limit for how many observations are held in total.
    current_idx : int
        current position in `buffer`; circular.
    current_size : int
        number of items in `buffer`; caps at `buffer_size`.
    state_dtype : type
        numpy datatype for states to be stored in.
    buffer : dict of {str : np.ndarray}
        all observations as dictionary of arrays.
    """
    def __init__(self, buffer_size, state_dtype):
        self.buffer_size = buffer_size
        self.current_idx = 0
        self.current_size = 0
        self.state_dtype = state_dtype

    def configure(self, agent):
        """Setup buffer, using shape information from action space and state pipeline.

        Parameters
        ----------
        agent : pavlov.Agent
            the Agent object that the replay buffer is associated with.
        """
        self.state_shape = agent.state_pipeline.out_dims
        self.action_space = agent.env.action_space
        self.action_type = self.action_space.__class__.__name__.lower()

        buffer = {}
        buffer['states'] = np.zeros([self.buffer_size] + self.state_shape,
                                    dtype=self.state_dtype)
        buffer['next_states'] = np.zeros([self.buffer_size] + self.state_shape,
                                         dtype=self.state_dtype)
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
        self.buffer = buffer

    def get_batch(self, batch_size):
        """Return random sample of observations from buffer for all keys, as dictionary.

        Parameters
        ----------
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

        Parameters
        ----------
        state : np.ndarray
            state to be added to buffer.
        action : np.ndarray
            model-compatible version of action to be added to buffer.
        reward : float
            reward to be added to buffer.
        next_state : np.ndarray
            next state to be added to buffer.
        done : bool
            done flag to be added to buffer.
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

    def __getitem__(self, key):
        return self.buffer[key]
