def get_action_type(space):
    s = str(type(space)).lower()
    return s[::-1][2:s[::-1].find('.')][::-1]


class ActionModelMismatchError(Exception):
    def __init__(self, action_type, model_type):
        message = f"Action of type {action_type} and model of type {model_type} do not correspond"
        super().__init__(message)
