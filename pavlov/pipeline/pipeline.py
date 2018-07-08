class Pipeline:
    """Container of a sequence of state transformation functions.

    Attributes
    ----------
    transformations : list of function
        the sequence of functions to be applied.
    out_dims : list of int
        the dimensions of the resulting, transformed state.
    """
    def __init__(self):
        self.transformations = []

    def configure(self, agent):
        """Calculate output shape of pipeline using agent's first env_state.

        Parameters
        ----------
        agent : pavlov.Agent
            Agent associated with Pipeline.
        """
        self.env = agent.env
        # out_dims must be list because it's being appended to a list later
        self.out_dims = list(self.transform(agent.env_state).shape)

    def add(self, transformation):
        """Add a transformation to the end of the pipeline.

        Parameters
        ----------
        transformation : function
            transformation to be added to the pipeline.
        """
        self.transformations.append(transformation)

    def transform(self, state):
        """Apply pipeline to incoming state.

        Parameters
        ----------
        state : np.ndarray
            state to pass through pipeline.

        Returns
        -------
        out : np.ndarray
            result of applying pipeline to state.
        """
        out = state
        for transformation in self.transformations:
            out = transformation(out, self.env)
        return out
