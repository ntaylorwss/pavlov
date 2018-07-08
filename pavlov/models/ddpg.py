import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, Flatten, Activation, Multiply
from .common import BaseModel


class DDPGModel(BaseModel):
    """A deep deterministic policy gradient model.

    Used for policy-based predictions in a continuous action space.
    Takes in a base topology, which is a headless Keras computation graph for state transformation.
    Adds layers to that as needed for actor and critic networks, including incorporating actions.

    Parameters
    ----------
    activation : str or keras.Activation
        final activation function just for actor model output.
    gamma : float
        discount factor for value function.
    tau : float
        mixing rate for target network update; how fast it follows main network.
    actor_optimizer : tf.train.Optimizer
        optimizer for actor network.
    critic_optimizer : keras.Optimizer
        optimizer for critic network.

    Attributes
    ----------
    prediction_type : {'policy', 'value'}
        indicating whether it's a policy-based or value-based model.
    gamma : float
        discount factor for value function.
    tau : float
        mixing rate for target network update; how fast it follows main network.
    actor_optimizer : tf.train.Optimizer
        optimizer for actor network.
    critic_optimizer : keras.Optimizer
        full optimizer object for critic network.
    action_space : gym.Space
        action space from environment.
    action_type : str
        lowercased class name of action space.
    actor : pavlov.ActorNetwork
        actor network with I/O of `state -> policy vector`.
    critic : pavlov.CriticNetwork
        critic network with I/O of `[state, action] -> value scalar`.

    Notes
    -----
        actor_optimizer must be of base type tf.train.Optimizer, meaning from the tf.train module.
        A Keras optimizer will not work for this particular parameter.
    """

    def __init__(self, topology, activation, gamma, tau,
                 actor_optimizer, critic_optimizer):
        super().__init__(topology)
        self.prediction_type = 'policy'
        self.gamma = gamma
        self.tau = tau
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def _configure_model(self):
        """Generate actor and critic models."""
        self.actor = ActorNetwork(self, self.session,
                                  self.topology.input, self.topology.output,
                                  self.activation, self.tau, self.actor_optimizer)
        self.critic = CriticNetwork(self, self.session,
                                    self.topology.input, self.topology.output,
                                    self.activation, self.tau, self.critic_optimizer)

    def has_nan(self):
        """Check both actor and critic models for nan."""
        return (any(np.isnan(np.sum(W)) for W in self.actor.model.get_weights())
                or any(np.isnan(np.sum(W)) for W in self.critic.model.get_weights()))

    def fit(self, states, actions, rewards, next_states, dones):
        """Generate target values, fit critic, then fit actor to critic gradients."""
        target_actions = self.actor.target_predict(next_states, single=False)
        target_q_values = self.critic.target_predict(next_states, target_actions,
                                                     single=False)
        targets = np.array([r if done else r + self.gamma * target_q
                   for r, target_q, done in zip(rewards, target_q_values, dones)])
        self.critic.fit(states, actions, targets)

        actions_for_gradients = self.actor.predict(states, single=False)
        gradients = self.critic.gradients(states, actions_for_gradients)
        self.actor.fit(states, gradients)

        self.actor.target_fit()
        self.critic.target_fit()

    def predict(self, state):
        """Use actor to produce policy vector for state."""
        return self.actor.predict(state)


class ActorNetwork:
    """Actor component of DDPG model; the network that predicts a policy vector based on the state.

    Parameters
    ----------
    model : pavlov.DDPGModel
        instance of DDPGModel that Actor is associated with.
    session : tf.Session
        tensorflow session that the models live in.
    base_input : keras.Layer
        state input layer for both networks.
    base_output : keras.Layer
        state-transforming intermediate output layer.
    activation : str or keras.Activation
        final activation function just for actor model output.
    tau : float
        mixing rate for target network update; how fast it follows main network.
    tf_optimizer : tf.train.Optimizer
        optimizer for model.
    """
    def __init__(self, model, session, base_input, base_output,
                 activation, tau, tf_optimizer):
        self.model = model
        self.session = session
        K.set_session(self.session)
        self.base_input = base_input
        self.base_output = base_output
        self.activation = activation
        self.tau = tau
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.action_grad_input = K.placeholder(shape=(None, self.action_dim))
        self.model_grad = tf.gradients(self.model.output, self.model.trainable_weights,
                                       -self.action_grad_input)
        grads = list(zip(self.model_grad, self.model.trainable_weights))
        self.optimize = tf_optimizer.apply_gradients(grads)

    def _create_model(self):
        """Add final output layer and return Model object.

        Returns
        -------
        keras.Model
        """
        if self.base_output.shape.ndims > 2:
            out = Flatten()(self.base_output)
        else:
            out = self.base_output

        # make function out of activation argument if it isn't one already
        if isinstance(self.activation, str):
            act = Activation(self.activation)
        else:
            act = self.activation
        # flatten action shape into vector
        out = act(Dense(np.prod(self.model.action_space.shape)))
        return Model(self.base_input, out)

    def fit(self, states, action_gradients):
        """Fit network based on state input and action gradients fed from critic.

        Parameters
        ----------
        states : np.ndarray
            states to fit to; input.
        action_gradients : np.ndarray
            gradients of critic model output with respect to the critic's action input; input.
        """
        self.session.run(self.optimize, feed_dict={
                self.model.input: states,
                self.action_grad_input: action_gradients})

    def target_fit(self):
        """Update target network towards main network according to tau."""
        W = [self.tau*aw + (1-self.tau)*tw
             for aw, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(W)

    def _predict(self, model, state, single):
        """General function to predict policy for state with a given model.

        Parameters
        ----------
        model : keras.Model
            either `self.model` or `self.target_model`.
        state : np.ndarray
            state input to predict for; could be single observation or batch.
        single : bool
            flag to indicate whether state input is single observation or batch.

        Returns
        -------
        pred : np.ndarray
            the prediction of the model for the given state.
        """
        if single:
            state = np.expand_dims(state, 0)
            pred = model.predict(state)[0]
            if len(self.action_space.shape) > 2:
                pred = pred.reshape(self.action_space.shape)
        else:
            pred = model.predict(state)
            if len(self.action_space.shape) > 2:
                pred = pred.reshape([pred.shape[0]] + list(self.action_space.shape))
        return pred

    def predict(self, state, single=True):
        """Predict a policy for the given state with the main model.

        Parameters
        ----------
        state : np.ndarray
            state input to predict for; could be single observation or batch.
        single : bool
            flag to indicate whether state input is single observation or batch
            (default is True, that it is a single observation).
        """
        return self._predict(self.model, state, single)

    def target_predict(self, state, single=True):
        """Predict a policy for the given state with the target model.

        Parameters
        ----------
        state : np.ndarray
            state input to predict for; could be single observation or batch.
        single : bool
            flag to indicate whether state input is single observation or batch
            (default is True, that it is a single observation).
        """
        return self._predict(self.target_model, state, single)


class CriticNetwork:
    """Critic component of DDPG model; the network that predicts a state-action value.

    Parameters
    ----------
    model : DDPGModel
        instance of DDPGModel that Critic is associated with.
    session : tf.Session
        tensorflow session that the models live in.
    base_input : keras.Layer
        state input layer for both networks.
    base_output : keras.Layer
        state-transforming intermediate output layer.
    tau : float
        mixing rate for target network update; how fast it follows main network.
    optimizer : keras.Optimizer
        full optimizer object.
    """
    def __init__(self, model, session, base_input, base_output, tau, optimizer):
        self.model = model
        self.session = session
        K.set_session(self.session)
        self.base_input = base_input
        self.base_output = base_output
        self.tau = tau
        self.optimizer = optimizer
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.action_grads = tf.gradients(self.model.output, self.model.input[1])
        self.session.run(tf.global_variables_initializer())

    def _create_model(self):
        """Add action input and final output layer and return Model object.

        Returns
        -------
        keras.Model
        """
        if self.base_output.shape.ndims > 2:
            base_out = Flatten()(self.base_output)
        else:
            base_out = self.base_output
        in_shape = (np.prod(self.action_space.shape),)
        action_in = Input(in_shape)
        out = Dense(in_shape, activation='relu')(base_out)
        out = Multiply()([out, action_in])
        out = Dense(1)(out)
        model = Model([self.base_input, action_in], out)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def fit(self, states, actions, targets):
        """Fit network to state and action input with targets from target network.

        Parameters
        ----------
        states : np.ndarray
            states to fit to; input.
        actions : np.ndarray
            actions to fit to; input.
        targets : np.ndarray
            targets to fit to; output.
        """
        self.model.train_on_batch([states, actions], targets)

    def target_fit(self):
        """Update target network towards main network according to `tau`."""
        W = [self.tau*cw + (1-self.tau)*tw
             for cw, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(W)

    def _predict(self, model, state, action, single):
        """General function to predict state-action value for given input with a given model.

        Parameters
        ----------
        model : keras.Model
            either `self.model` or `self.target_model`.
        state : np.ndarray
            state input to predict for; could be single observation or batch.
        action : np.ndarray
            action input to predict for; could be single observation or batch.
        single : bool
            flag to indicate whether state input is single observation or batch.

        Returns
        -------
        pred : np.ndarray
            the prediction of the model for the given state.
        """
        if single:
            inputs = [np.expand_dims(state, 0), np.expand_dims(action, 0)]
            return model.predict(inputs)[0]
        else:
            inputs = [state, action]
            return model.predict(inputs)

    def predict(self, state, action, single=True):
        """Predict a policy for the given state and action with the main model.

        Parameters
        ----------
        model : keras.Model
            either `self.model` or `self.target_model`.
        state : np.ndarray
            state input to predict for; could be single observation or batch.
        action : np.ndarray
            action input to predict for; could be single observation or batch.
        single : bool
            flag to indicate whether state input is single observation or batch.

        Returns
        -------
        pred : np.ndarray
            the prediction of the model for the given state.
        """
        return self._predict(self.model, state, single)

    def target_predict(self, state, action, single=True):
        """Predict a policy for the given state and action with the target model.

        Parameters
        ----------
        model : keras.Model
            either `self.model` or `self.target_model`.
        state : np.ndarray
            state input to predict for; could be single observation or batch.
        action : np.ndarray
            action input to predict for; could be single observation or batch.
        single : bool
            flag to indicate whether state input is single observation or batch.

        Returns
        -------
        pred : np.ndarray
            the prediction of the model for the given state.
        """
        return self._predict(self.target_model, state, action, single)

    def gradients(self, states, actions):
        """Get gradient of action with respect to the model parameters for given state and action.

        Parameters
        ----------
        states : np.ndarray
            state input to predict for.
        actions : np.ndarray
            action input to predict for.

        Returns
        -------
        np.ndarray
        """
        return self.session.run(self.action_grads, feed_dict={
                    self.model.input[0]: states,
                    self.model.input[1]: actions})[0]
