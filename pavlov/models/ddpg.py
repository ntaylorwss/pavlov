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

    Additional parameters:
        activation (str/keras.Activation): final activation function just for actor model output.
        gamma (float): discount factor for value function.
        tau (float): mixing rate for target network update; how fast it follows main network.
        actor_optimizer (tf.train.Optimizer): full optimizer object for actor network.
                                              MUST be tensorflow type, not Keras, unfortunately.
        critic_optimizer (keras.Optimizer): full optimizer object for critic network.

    Additional member variables:
        prediction_type (str): indicating whether it's a policy-based or value-based model.
                               defined by child class;
                               parent's member is a placeholder.
                               options: policy, value.
        gamma (float): discount factor for value function.
        tau (float): mixing rate for target network update; how fast it follows main network.
        actor_optimizer (tf.train.Optimizer): full optimizer object for actor network.
                                              MUST be tensorflow type, not Keras, unfortunately.
        critic_optimizer (keras.Optimizer): full optimizer object for critic network.
        action_space (gym.spaces.*): whole action space object from environment.
        action_type (str): class name of action space.
        actor (pavlov.ActorNetwork): actor network. I/O: state -> policy vector.
        critic (pavlov.CriticNetwork): critic network. I/O: [state, action] -> value scalar.
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
        self.actor = ActorNetwork(self, self.session,
                                  self.topology.input, self.topology.output,
                                  self.activation, self.tau, self.actor_optimizer)
        self.critic = CriticNetwork(self, self.session,
                                    self.topology.input, self.topology.output,
                                    self.activation, self.tau, self.critic_optimizer)

    def fit(self, states, actions, rewards, next_states, dones):
        """Fit model to batch from experience."""
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
        """Return policy vector for given state from actor."""
        return self.actor.predict(state)


class ActorNetwork:
    """Actor component of DDPG model; the network that predicts a policy vector based on the state.

    Parameters:
        model (DDPGModel): instance of DDPGModel that Actor is associated with.
        session (tf.Session): tensorflow session that the models live in.
        base_input (keras.Layer): state input layer for both networks.
        base_output (keras.Layer): state-transforming intermediate output layer.
        activation (str/keras.Activation): final activation function just for actor model output.
        tau (float): mixing rate for target network update; how fast it follows main network.
        tf_optimizer (tf.train.Optimizer): full optimizer object.
                                           MUST be tensorflow type, not Keras, unfortunately.
    """
    def __init__(self, model, session, base_input, base_output,
                 activation, tau, tf_optimizer):
        self.model = model
        self.session = session
        K.set_session(self.session)
        self.base_input, = base_input
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
        """Extend `base_topology` to produce a policy output."""
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
        """Fit network based on state input and action gradients fed from critic."""
        self.session.run(self.optimize, feed_dict={
                self.model.input: states,
                self.action_grad_input: action_gradients})

    def target_fit(self):
        """Update target network towards main network according to `tau`."""
        W = [self.tau*aw + (1-self.tau)*tw
             for aw, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(W)

    def _predict(self, model, state, single):
        """General function to predict policy for state with a given model.

        Parameters:
            model (keras.Model): either `self.model` or `self.target_model`.
            state (np.array): state input to predict; could be single observation or batch.
            single (bool): flag to indicate whether state input is single observation or batch.
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
        """Apply `_predict` with `self.model`."""
        return self._predict(self.model, state, single)

    def target_predict(self, state, single=True):
        """Apply `_predict` with `self.target_model`."""
        return self._predict(self.target_model, state, single)


class CriticNetwork:
    """Critic component of DDPG model; the network that predicts a state-action value.

    Parameters:
        model (DDPGModel): instance of DDPGModel that Critic is associated with.
        session (tf.Session): tensorflow session that the models live in.
        base_input (keras.Layer): state input layer for both networks.
        base_output (keras.Layer): state-transforming intermediate output layer.
        tau (float): mixing rate for target network update; how fast it follows main network.
        optimizer (keras.Optimizer): full optimizer object.
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
        """Extend `base_topology` to produce an action-value output (action as input)."""
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
        """Fit network based on state input, action from actor, targets from target critic."""
        self.model.train_on_batch([states, actions], targets)

    def target_fit(self):
        """Update target network towards main network according to `tau`."""
        W = [self.tau*cw + (1-self.tau)*tw
             for cw, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(W)

    def _predict(self, model, state, action, single):
        """General function to predict value for state and action with given model.

        Parameters:
            model (keras.Model): either `self.model` or `self.target_model`.
            state (np.array): state input to predict for; could be single observation or batch.
            action (np.array): action input to predict for; could be single observation or batch.
            single (bool): flag to indicate whether state/action input is single observation or batch.
        """
        if single:
            inputs = [np.expand_dims(state, 0), np.expand_dims(action, 0)]
            return model.predict(inputs)[0]
        else:
            inputs = [state, action]
            return model.predict(inputs)

    def predict(self, state, action, single=True):
        """Apply `_predict` with `self.model`."""
        return self._predict(self.model, state, single)

    def target_predict(self, state, action, single=True):
        """Apply `_predict` with `self.target_model`."""
        return self._predict(self.target_model, state, action, single)

    def gradients(self, states, actions):
        """Get gradient of model with respect to actions, to be passed to actor."""
        return self.session.run(self.action_grads, feed_dict={
                    self.model.input[0]: states,
                    self.model.input[1]: actions})[0]
