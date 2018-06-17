import numpy as np

def twoDcont_to_index():
    def _twoDcont_to_index(pred, env):
        action = (int(pred[0] * env.n_rows), int(pred[1] * env.n_cols))
        return action[0] * env.n_cols + action[1]
    return _twoDcont_to_index


def twoDindex_to_index():
    def _twoDindex_to_index(pred, env):
        return pred[0] * env.n_cols + pred[1]
    return _twoDindex_to_index


def argmax_scalar():
    def _argmax_scalar(pred, env):
        return np.random.choice(np.flatnonzero(pred == pred.max()))
    return _argmax_scalar


def scalar_to_onehot():
    def _scalar_to_onehot(pred, env):
        out = np.zeros(env.action_space.n)
        out[pred] = 1
        return out
    return _scalar_to_onehot
