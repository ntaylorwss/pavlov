import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate, Activation
from keras.optimizers import Adam


class DQNModel:
    def __init__(self, base_topology, action_dim, gamma, tau, alpha):
        self.session = tf.Session()
        K.set_session(self.session)
        self.base_input, self.base_output = base_topology
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.model = self._create_model()
        self.target_model = self._create_model()

    def _create_model(self):
        action_in = Input((self.action_dim,))
        if self.base_output.shape.ndims > 2:
            base_out = Flatten()(self.base_output)
        else:
            base_out = self.base_output
        out = concatenate([base_out, action_in])
        out = Dense(1)(out)
        model = Model([self.base_input, action_in], out)
        opt = Adam(lr=self.alpha)
        model.compile(loss='mse', optimizer=opt)
        return model

    def fit(self, states, actions, rewards, next_states, dones):
        max_next_q = np.max(self.target_predict(next_states, single=False), axis=1)
        targets = rewards + dones * self.gamma * max_next_q
        self.model.train_on_batch([states, actions], targets)

    def target_fit(self):
        W = [self.tau*cw + (1-self.tau)*tw
             for cw, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(W)

    def _predict(self, model, state, single):
        if single:
            state = np.expand_dims(state, 0)
        out = np.hstack([model.predict([state, np.zeros((state.shape[0],
                                                        self.action_dim)) + a])
                         for a in np.eye(self.action_dim)])
        return out[0] if single else out

    def predict(self, state, single=True):
        return self._predict(self.model, state, single)

    def target_predict(self, state, single=True):
        return self._predict(self.target_model, state, single)
