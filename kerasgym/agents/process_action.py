def twoDcont_to_index():
    def _twoDcont_to_index(pred, env):
        action = (int(pred[0] * env.n_rows), int(pred[1] * env.n_cols))
        return action[0] * env.n_cols + action[1]
    return _twoDcont_to_index


def twoDindex_to_index():
    def _twoDindex_to_index(pred, env):
        return pred[0] * env.n_cols + pred[1]
    return _twoDindex_to_index


def argmax():
    def _argmax(pred, env):
        return max(range(len(pred)), key = lambda i: pred[i])
    return _argmax


def identity():
    def _identity(pred, env):
        return pred
    return _identity
