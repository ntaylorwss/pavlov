import numpy as np


def raw_array():
    def _raw_array(env, state):
        return np.reshape(np.array(state), (env.n_rows, env.n_cols))
    return _raw_array


def add_channel_dim():
    def _add_channel_dim(env, state):
        return np.expand_dims(state, 3)
    return _add_channel_dim


def one_hot():
    def _one_hot(env, state):
        return (np.arange(-1, 9) == state[..., None]) * 1
    return _one_hot
