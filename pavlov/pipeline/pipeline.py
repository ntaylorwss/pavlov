class Pipeline:
    """Container of a sequence of state transformation functions.

    Parameters: none.

    Member variables:
        transformations (list[functions]): the sequence of functions to be applied.
        out_dims (list[int]): the dimensions of the resulting, transformed state.
    """
    def __init__(self):
        self.transformations = []

    def configure(self, agent):
        """Calculate output shape of pipeline using agent's first env_state."""
        self.env = agent.env
        # out_dims must be list because it's being appended to a list later
        self.out_dims = list(self.transform(agent.env_state).shape)

    def add(self, transformation):
        """Add a transformation to the end of the pipeline."""
        self.transformations.append(transformation)

    def transform(self, state):
        """Apply pipeline to incoming state."""
        out = state
        for transformation in self.transformations:
            out = transformation(out, self.env)
        return out
