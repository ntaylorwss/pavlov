class ActionModelMismatchError(Exception):
    """Throw when action space and prediction type don't align.
    Example: value functions cannot be used directly in a continuous environment."""
    def __init__(self, action_type, model_type):
        message = f"Action of type {action_type} and model of type {model_type} do not correspond"
        super().__init__(message)
