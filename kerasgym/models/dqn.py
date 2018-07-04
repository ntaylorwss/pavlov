import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Multiply
from .. import util
from .common import BaseModel


class DQNModel(BaseModel):
    """A deep Q network model.

    Used for value-based predictions in a discrete action space.
    Takes in a base topology, which is a headless Keras computation graph for state transformation.
    Adds layers to that as needed for action incorporation, produces state-action values.

    Parameters:
        gamma (float): discount factor for Q Learning.
        tau (float): mixing rate for target network update; how fast it follows main network.
        optimizer (keras.Optimizer): full optimizer object.

    Member variables:
        prediction_type (str): indicating whether it's a policy-based or value-based model.
                               defined by child class;
                               parent's member is a placeholder.
                               options: policy, value.
        gamma (float): discount factor for Q Learning.
        tau (float): mixing rate for target network update; how fast it follows main network.
        optimizer (keras.Optimizer): full optimizer object.
    """
    def __init__(self, topology, gamma, tau, optimizer):
        super().__init__(topology)
        self.prediction_type = 'value'
        self.gamma = gamma
        self.tau = tau
        self.optimizer = optimizer

    def _configure_model(self):
        self.model = self._topology_to_model()
        self.target_model = self._topology_to_model()

    def _topology_to_model(self):
        """Extend `self.topology` to produce an action-value output (action as input)."""
        if self.topology.output.shape.ndims > 2:
            base_out = Flatten()(self.topology.output)
        else:
            base_out = self.topology.output

        if self.action_type == 'discrete':
            action_in = Input((self.action_space.n,))
            out = Dense(self.action_space.n)(base_out)
            out = Multiply()([out, action_in])
            model = Model([self.topology.input, action_in], out)
        elif self.action_type == 'multidiscrete':
            action_ins = [Input((n,)) for n in self.action_space.nvec]
            outs = [Dense(n)(base_out) for n in self.action_space.nvec]
            outs = [Multiply()([out, action_in]) for out, action_in in zip(outs, action_ins)]
            model = Model([self.topology.input] + action_ins, outs)
        elif self.action_type == 'multibinary':
            action_ins = [Input((2,)) for _ in range(self.action_space.n)]
            outs = [Dense(2)(base_out) for _ in range(self.action_space.n)]
            outs = [Multiply()([out, action_in]) for out, action_in in zip(outs, action_ins)]
            model = Model([self.topology.input] + action_ins, outs)

        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def fit(self, states, actions, rewards, next_states, dones):
        """Fit network to batch of observations from replay buffer."""
        max_next_q = np.max(self.target_predict(next_states, single=False), axis=1)
        targets = np.expand_dims(rewards + (1 - dones) * self.gamma * max_next_q, -1)
        targets = targets * actions  # makes it one-hot, value in the place of the action
        self.model.train_on_batch([states, actions], targets)
        self.target_fit()

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
        state = np.expand_dims(state, 0) if single else state
        if self.action_type == 'discrete':
            if action and single:
                action = np.expand_dims(action, 0)
            elif single:
                action = np.ones((1, self.action_space.n))
            else:
                action = np.ones((state.shape[0], self.action_space.n))
            model_in = [state, action]
        elif self.action_type == 'multidiscrete':
            if action and single:
                action = [np.expand_dims(a, 0) for a in action]
            elif single:
                action = [np.ones((1, n)) for n in self.action_space.nvec]
            else:
                action = [np.ones((state.shape[0], n)) for n in self.action_space.nvec]
            model_in = [state] + action
        elif self.action_type == 'multibinary':
            if action and single:
                action = [np.expand_dims(a, 0) for a in action]
            elif single:
                action = [np.ones((1, 2)) for _ in range(self.action_space.n)]
            else:
                action = [np.ones((state.shape[0], 2)) for _ in range(self.action_space.n)]
            model_in = [state] + action
        return model.predict(model_in)

    def predict(self, state, action=None, single=True):
        """Apply `_predict` with `self.model`."""
        return self._predict(self.model, state, action, single)

    def target_predict(self, state, action=None, single=True):
        """Apply `_predict` with `self.target_model`."""
        return self._predict(self.target_model, state, action, single)
