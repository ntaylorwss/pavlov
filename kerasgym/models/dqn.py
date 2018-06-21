import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Activation
from keras.layers import concatenate, Add, Multiply


class DQNModel:
    def __init__(self, action_dim, base_topology, gamma, tau, optimizer):
        self.pred_type = 'value'
        self.session = tf.Session()
        K.set_session(self.session)
        self.base_input, self.base_output = base_topology
        self.in_shape = self.base_input.shape.as_list()[1:]
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.optimizer = optimizer
        self.model = self._create_model()
        self.target_model = self._create_model()

    def _create_model(self):
        action_in = Input((self.action_dim,))
        if self.base_output.shape.ndims > 2:
            base_out = Flatten()(self.base_output)
        else:
            base_out = self.base_output
        out = Dense(self.action_dim)(self.base_output)
        out = Multiply()([out, action_in])
        model = Model([self.base_input, action_in], out)

        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def target_fit(self):
        W = [self.tau*cw + (1-self.tau)*tw
             for cw, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(W)

    def fit(self, states, actions, rewards, next_states, dones):
        max_next_q = np.max(self.target_predict(next_states, single=False), axis=1)
        targets = np.expand_dims(rewards + (1 - dones) * self.gamma * max_next_q, -1)
        targets = targets * actions # makes it one-hot, value in the place of the action
        self.model.train_on_batch([states, actions], targets)
        self.target_fit()

    def _predict(self, model, state, action, single):
        n_actions = model.input[1].shape[1].value
        if single:
            if action:
                action = np.expand_dims(action, 0)
            else:
                action = np.ones((1, n_actions))
            state = np.expand_dims(state, 0)
            return model.predict([state, action])
        else:
            if action is None:
                action = np.ones((state.shape[0], n_actions))
            return model.predict([state, action])

    def predict(self, state, action=None, single=True):
        return self._predict(self.model, state, action, single)

    def target_predict(self, state, action=None, single=True):
        return self._predict(self.target_model, state, action, single)
