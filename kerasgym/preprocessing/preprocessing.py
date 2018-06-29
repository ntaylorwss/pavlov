import numpy as np
import collections.deque
from skimage.transform import resize


def raw_array():
    def _raw_array(state, env):
        return np.reshape(np.array(state), (env.n_rows, env.n_cols))
    return _raw_array


def add_channel_dim():
    def _add_channel_dim(state, env):
        return np.expand_dims(state, 3)
    return _add_channel_dim


def one_hot():
    def _one_hot(state, env):
        return (np.arange(-1, 9) == state[..., None]) * 1
    return _one_hot


def rgb_to_binary():
    def _rgb_to_binary(state, env):
        return (state.max(axis=2) > 0).astype(int)[:,:,np.newaxis]
    return _rgb_to_binary


def downsample(new_shape):
    def _downsample(state, env):
        return resize(state, new_shape)
    return _downsample


class stack_consecutive:
    def __init__(self, n_frames=4):
        self.previous_frames = collections.deque([], maxlen=n_frames-1)

    def __call__(self, state, env):
        n_missing = self.previous_frames.maxlen - len(self.previous_frames)
        if n_missing > 0:
            out = np.dstack([np.zeros(state.shape)]*n_missing + \
                             list(self.previous_frames) + [state])
        else:
            out = np.dstack(list(self.previous_frames) + [state])
        self.previous_frames.append(state)
        return out


class combine_consecutive:
    def __init__(self, n_previous_frames=1, fun='max'):
        self.previous_frames = collections.deque([], maxlen=n_previous_frames)
        self.fun = fun

    def __call__(self, state, env):
        if len(self.previous_frames) == 0:
            out = state
        else:
            out = np.dstack(list(self.previous_frames) + [state])
            if self.fun == 'max':    out = out.max(axis=-1)
            elif self.fun == 'min':  out = out.min(axis=-1)
            elif self.fun == 'mean': out = out.mean(axis=-1)
            elif self.fun == 'diff': out = out[..., -1] - out[..., 0]
            else:                    out = out.max(axis=-1)
        self.previous_frames.append(state)
        return out
