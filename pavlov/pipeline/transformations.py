import numpy as np
import collections
from skimage.transform import resize


def to_array():
    """Convert state to numpy array data type."""
    def _to_array(state, env):
        return np.array(state)
    return _to_array


def reshape_array(new_shape):
    """Reshape state array to given new shape.

    Parameters
    ----------
    new_shape : tuple
        target shape of array.
    """
    def _reshape_array(state, env):
        return np.reshape(state, new_shape)
    return _reshape_array


def add_dim(axis=-1):
    """Add dimension to state array.

    Parameters
    ----------
    pos : int
        axis along which to inject new dimension (default is -1, which means last).
    """
    def _add_dim(state, env):
        return np.expand_dims(state, axis)
    return _add_dim


def one_hot(max_val, min_val=0):
    """One-hot encode entire state array according to defined range of integer values.

    Parameters
    ----------
    max_val : int
        maximum categorical value in state.
    min_val : int
        minimum categorical value in state (default is 0).
    """
    def _one_hot(state, env):
        return (np.arange(min_val, max_val+1) == state[..., None]) * 1
    return _one_hot


def rgb_to_grey(method='luminosity'):
    """Convert 3D RGB image to 2D greyscale image using one of 3 algorithms.

    Parameters
    ----------
    method : {'luminosity', 'average', 'lightness'}
        algorithm for averaging.
    """
    def _rgb_to_grey(state, env):
        if method == 'lightness':
            return (state.max(axis=2) + state.min(axis=2)) / 2.
        elif method == 'average':
            return state.mean(axis=2)
        elif method == 'luminosity':
            return np.matmul(state, np.array([0.21, 0.72, 0.07]))
        else:
            raise ValueError("Invalid averaging method for rgb_to_grey.")
    return _rgb_to_grey


def rgb_to_binary():
    """Converts 3D RGB image to 3D binary with 1 channel, where a 1 indicates a nonzero value."""
    def _rgb_to_binary(state, env):
        return (state.max(axis=2) > 0).astype(int)[:, :, np.newaxis]
    return _rgb_to_binary


# following two are separate functions for readability; more clearly self-documenting this way
def downsample(new_shape):
    """Resize input image to smaller shape via interpolation.

    Parameters
    ----------
    new_shape : tuple
        target shape of array after resizing.
    """
    def _downsample(state, env):
        if any(new_shape[i] > state.shape[i] for i in range(len(new_shape))):
            raise ValueError("New shape is larger than current in at least one dimension.")
        return resize(state, new_shape)
    return _downsample


def upsample(new_shape):
    """Resize input image to larger shape via interpolation.

    Parameters
    ----------
    new_shape : tuple
        target shape of array after resizing.
    """
    def _upsample(state, env):
        if any(new_shape[i] < state.shape[i] for i in range(len(new_shape))):
            raise ValueError("New shape is smaller than current in at least one dimension.")
        return resize(state, new_shape)
    return _upsample


# some functions are classes because they require storing information, such as previous states
class stack_consecutive:
    """Transform state into time-series of `n_states` recent states, along new dimension.

    Parameters:
        n_states (int): number of states to include in time-series, including current.
    """
    def __init__(self, n_states=4):
        self.previous_states = collections.deque([], maxlen=n_states-1)

    def __call__(self, state, env):
        n_missing = self.previous_states.maxlen - len(self.previous_states)
        if n_missing > 0:
            # fill missing with zeros
            filler = [np.zeros(state.shape)] * n_missing
            out = np.stack([state] + filler + list(self.previous_states))
        else:
            out = np.stack([state] + list(self.previous_states))
        self.previous_states.appendleft(state)
        return out


class combine_consecutive:
    """Combine previous `n_states` states by a statistic; retains original shape.

    Parameters:
        n_states (int): number of states to use for statistic, including current.
        fun (str): function to combine series of states with.
                   options: max, min, mean, diff.
                   default: max.
    """
    def __init__(self, n_states=2, fun='max'):
        self.previous_states = collections.deque([], maxlen=n_states-1)
        self.fun = fun

    def __call__(self, state, env):
        if len(self.previous_states) == 0:
            out = state
        else:
            out = np.dstack(list(self.previous_states) + [state])
            if self.fun == 'max':
                out = out.max(axis=-1)
            elif self.fun == 'min':
                out = out.min(axis=-1)
            elif self.fun == 'mean':
                out = out.mean(axis=-1)
            elif self.fun == 'diff':
                out = out[..., -1] - out[..., 0]
            else:
                out = out.max(axis=-1)
        self.previous_states.append(state)
        return out
