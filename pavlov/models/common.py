import tensorflow as tf
import keras.backend as K


class BaseModel:
    """Base class for all kinds of reinforcement learning models.

    All models should take in some form of feature-extracting topology,
    which is how the user defines the architecture of the model.
    From there this model should configure the input and output according
    to the other components it's connected to via the Agent.
    The public interface should also be common among models, so that is
    defined here as well.

    Common parameters:
        topology (pavlov.Topology): feature-extracting Keras model graph object
                                      defining the body of the model.
    Common member variables:
        topology (pavlov.Topology): feature-extracting Keras model graph object
                                      defining the body of the model.
        prediction_type (str): indicating whether it's a policy-based or value-based model.
                               defined by child class;
                               parent's member is a placeholder.
                               options: policy, value.
        session (tf.Session): tensorflow session that the model(s) will live in.
    """
    def __init__(self, topology):
        self.topology = topology
        self.prediction_type = None
        self.session = tf.Session()
        K.set_session(self.session)

    def _configure_model(self):
        """Generate necessary models, probably finishing off `self.topology` to do so."""
        pass

    def configure(self, agent):
        """Associate model with action space from environment; convert topology to model."""
        self.action_space = agent.env.action_space
        self.action_type = self.action_space.__class__.__name__.lower()
        self.topology.configure(agent)
        self._configure_model()

    def fit(self, states, actions, rewards, next_states, dones):
        """Fit model to batch from experience. May or may not use all inputs."""
        pass

    def predict(self, state):
        """Return model output for given state input. May be policy or value."""
        pass
