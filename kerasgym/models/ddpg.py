import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, Flatten, Activation, Add

# TODO: like DQN, make this model work with any action space by drowning it in if statements.

class DDPGModel:
    def __init__(self, action_dim, base_topology,
                 actor_activation, gamma, tau, actor_alpha, critic_alpha):
        self.pred_type = 'policy'
        self.session = tf.Session()
        K.set_session(self.session)
        self.gamma = gamma
        self.actor = ActorNetwork(self.session, base_topology, action_dim,
                                  actor_activation, tau, actor_alpha)
        self.critic = CriticNetwork(self.session, base_topology, action_dim,
                                    tau, critic_alpha)
        self.in_shape = self.actor.in_shape
        self.action_dim = self.actor.action_dim

    def fit(self, states, actions, rewards, next_states, dones):
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
        return self.actor.predict(state)


class ActorNetwork:
    def __init__(self, session, base_topology,
                 action_dim, activation, tau, alpha):
        self.session = session
        K.set_session(self.session)
        self.action_dim = action_dim
        self.activation = activation
        self.tau = tau
        self.alpha = alpha
        self.base_input, self.base_output = base_topology
        self.in_shape = self.base_input.shape.as_list()[1:]
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.action_grad_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.model_grad = tf.gradients(self.model.output, self.model.trainable_weights,
                                       -self.action_grad_input)
        grads = list(zip(self.model_grad, self.model.trainable_weights))
        self.optimize = tf.train.AdamOptimizer(self.alpha).apply_gradients(grads)
        self.session.run(tf.global_variables_initializer())

    def _create_model(self):
        if self.base_output.shape.ndims > 2:
            out = Flatten()(self.base_output)
        else:
            out = self.base_output
        out = Activation(self.activation)(Dense(self.action_dim)(out))
        return Model(self.base_input, out)

    def fit(self, states, action_gradients):
        self.session.run(self.optimize, feed_dict={
                self.model.input: states,
                self.action_grad_input: action_gradients})

    def target_fit(self):
        W = [self.tau*aw + (1-self.tau)*tw
             for aw, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(W)

    def _predict(self, model, state, single):
        if single:
            state = np.expand_dims(state, 0)
            return model.predict(state)[0]
        else:
            return model.predict(state)

    def predict(self, state, single=True):
        return self._predict(self.model, state, single)

    def target_predict(self, state, single=True):
        return self._predict(self.target_model, state, single)


class CriticNetwork:
    def __init__(self, session, base_topology, action_dim, tau, alpha):
        self.session = session
        K.set_session(self.session)
        self.action_dim = action_dim
        self.tau = tau
        self.alpha = alpha
        self.base_input, self.base_output = base_topology
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.action_grads = tf.gradients(self.model.output, self.model.input[1])
        self.session.run(tf.global_variables_initializer())

    def _create_model(self):
        action_in = Input((self.action_dim,))
        if self.base_output.shape.ndims > 2:
            base_out = Flatten()(self.base_output)
        else:
            base_out = self.base_output
        # add the state and action by first making them same shape
        base_out = Dense(self.action_dim)(base_out)
        action_mid = Dense(self.action_dim)(action_in)
        base_and_action = Add()([base_out, action_mid])
        out = Dense(1)(base_and_action)
        model = Model([self.base_input, action_in], out)
        opt = Adam(lr=self.alpha)
        model.compile(loss='mse', optimizer=opt)
        return model

    def fit(self, states, actions, targets):
        self.model.train_on_batch([states, actions], targets)

    def target_fit(self):
        W = [self.tau*cw + (1-self.tau)*tw
             for cw, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(W)

    def _predict(self, model, state, action, single):
        if single:
            inputs = [np.expand_dims(state, 0), np.expand_dims(action, 0)]
            return model.predict(inputs)[0]
        else:
            inputs = [state, action]
            return model.predict(inputs)

    def predict(self, state, action, single=True):
        return self._predict(self.model, state, single)

    def target_predict(self, state, action, single=True):
        return self._predict(self.target_model, state, action, single)

    def gradients(self, states, actions):
        return self.session.run(self.action_grads, feed_dict={
                    self.model.input[0]: states,
                    self.model.input[1]: actions})[0]
