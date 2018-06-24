def get_action_type(space):
    s = str(type(space)).lower()
    return s[::-1][2:s[::-1].find('.')][::-1]
