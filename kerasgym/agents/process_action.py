def twoDcont_to_index():
    def _twoDcont_to_index(env, pred):
        action = (int(pred[0] * env.n_rows), int(pred[1] * env.n_cols))
        return action[0] * env.n_cols + action[1]
    return _twoDcont_to_index


def twoDindex_to_index():
    def _twoDindex_to_index(env, pred):
        return pred[0] * env.n_cols + pred[1]
    return _twoDindex_to_index


def identity():
    def _identity(env, pred):
        return pred
    return _identity
