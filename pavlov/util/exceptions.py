class ActionModelMismatchError(Exception):
    """Throw when action space and prediction type don't align.

    Example: value functions cannot be used directly in a continuous environment.
    """
    def __init__(self, action_type, model_type):
        message = "Action of type {} and model of type {} do not correspond".format(
                    action_type, model_type)
        super().__init__(message)
