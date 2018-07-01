import numpy as np
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, Flatten, Activation, Multiply
from .. import util


class DDPGModel:
    def __init__(self, base_topology, activation, gamma, tau,
                 actor_optimizer, critic_optimizer):
        self.pred_type = 'policy'
        self.session = tf.Session()
        K.set_session(self.session)
        self.base_input, self.base_output = base_topology
        self.in_shape = base_topology[0].shape.as_list()[1:]
        self.gamma = gamma
        self.tau = tau
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def configure(self, action_space):
        self.action_space = action_space
        self.action_type = util.get_action_type(action_space)
        self.actor = ActorNetwork(self, self.session, self.base_input, self.base_output,
                                  self.activation, self.tau, self.actor_optimizer)
        self.critic = CriticNetwork(self, self.session, self.base_input, self.base_output,
                                    self.activation, self.tau, self.critic_optimizer)

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
    def __init__(self, model, session, base_input, base_output,
                 activation, tau, alpha, tf_optimizer):
        self.model = model
        self.session = session
        K.set_session(self.session)
        self.base_input, = base_input
        self.base_output = base_output
        self.activation = activation
        self.tau = tau
        self.alpha = alpha
        self.in_shape = self.base_input.shape.as_list()[1:]
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.action_grad_input = K.placeholder(shape=(None, self.action_dim))
        self.model_grad = tf.gradients(self.model.output, self.model.trainable_weights,
                                       -self.action_grad_input)
        grads = list(zip(self.model_grad, self.model.trainable_weights))
        self.optimize = tf_optimizer.apply_gradients(grads)

    def _create_model(self):
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
            pred = model.predict(state)[0]
            if len(self.action_space.shape) > 2:
                pred = pred.reshape(self.action_space.shape)
        else:
            pred = model.predict(state)
            if len(self.action_space.shape) > 2:
                pred = pred.reshape([pred.shape[0]] + list(self.action_space.shape))
        return pred

    def predict(self, state, single=True):
        return self._predict(self.model, state, single)

    def target_predict(self, state, single=True):
        return self._predict(self.target_model, state, single)


class CriticNetwork:
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
