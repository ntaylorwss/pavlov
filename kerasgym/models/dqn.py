import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Multiply
from .. import util


class DQNModel:
    def __init__(self, base_topology, gamma, tau, optimizer):
        self.pred_type = 'value'
        self.session = tf.Session()
        K.set_session(self.session)
        self.base_input, self.base_output = base_topology
        self.in_shape = self.base_input.shape.as_list()[1:]
        self.gamma = gamma
        self.tau = tau
        self.optimizer = optimizer

    def configure(self, action_space):
        """Gets the action space, finishes off model based on that."""
        self.action_space = action_space
        self.action_type = action_space.__class__.__name__.lower()
        self.model = self._create_model()
        self.target_model = self._create_model()

    def _create_model(self):
        if self.base_output.shape.ndims > 2:
            base_out = Flatten()(self.base_output)
        else:
            base_out = self.base_output

        if self.action_type == 'discrete':
            action_in = Input((self.action_space.n,))
            out = Dense(self.action_space.n)(base_out)
            out = Multiply()([out, action_in])
            model = Model([self.base_input, action_in], out)
        elif self.action_type == 'multidiscrete':
            action_ins = [Input((n,)) for n in self.action_space.nvec]
            outs = [Dense(n)(base_out) for n in self.action_space.nvec]
            outs = [Multiply()([out, action_in]) for out, action_in in zip(outs, action_ins)]
            model = Model([self.base_input] + action_ins, outs)
        elif self.action_type == 'multibinary':
            action_ins = [Input((2,)) for _ in range(self.action_space.n)]
            outs = [Dense(2)(base_out) for _ in range(self.action_space.n)]
            outs = [Multiply()([out, action_in]) for out, action_in in zip(outs, action_ins)]
            model = Model([self.base_input] + action_ins, outs)

        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def target_fit(self):
        W = [self.tau*cw + (1-self.tau)*tw
             for cw, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(W)

    def fit(self, states, actions, rewards, next_states, dones):
        max_next_q = np.max(self.target_predict(next_states, single=False), axis=1)
        targets = np.expand_dims(rewards + (1 - dones) * self.gamma * max_next_q, -1)
        targets = targets * actions  # makes it one-hot, value in the place of the action
        self.model.train_on_batch([states, actions], targets)
        self.target_fit()

    def _predict(self, model, state, action, single):
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
        return self._predict(self.model, state, action, single)

    def target_predict(self, state, action=None, single=True):
        return self._predict(self.target_model, state, action, single)
