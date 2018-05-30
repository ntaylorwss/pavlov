import numpy as np
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Dense, Flatten


class CNNPolicyGradient:
    """Main purpose of this will be to hold actor, critic, action_input."""
    def __init__(self, in_shape, conv_layer_sizes, fc_layer_sizes, action_dim,
                 kernel_sizes, strides, mid_activation, final_activation):
        self.in_shape = in_shape
        self.conv_layer_sizes = conv_layer_sizes
        self.fc_layer_sizes = fc_layer_sizes
        self.action_dim = action_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.mid_activation = mid_activation
        self.final_activation = final_activation
        self.actor = self._create_actor_model()
        self.critic = self._create_critic_model()
        self.action_input = self.critic.input[1]

    def _create_actor_model(self):
        in_layer = Input(shape=self.in_shape)
        out = in_layer
        for l_size, k_size, stride in zip(self.conv_layer_sizes, self.kernel_sizes, self.strides):
            out = Conv2D(filters=l_size, kernel_size=k_size, padding='same',
                         strides=stride, activation=self.mid_activation)(out)
        out = Flatten()(out)
        for l_size in self.fc_layer_sizes:
            out = Dense(l_size, activation=self.mid_activation)(out)
        out = Dense(self.action_dim, activation=self.final_activation)(out)
        model = Model(in_layer, out)
        return model

    def _create_critic_model(self):
        state_in = Input(shape=self.in_shape)
        action_in = Input(shape=(self.action_dim,))
        out = state_in
        for l_size, k_size, stride in zip(self.conv_layer_sizes, self.kernel_sizes, self.strides):
            out = Conv2D(filters=l_size, kernel_size=k_size, padding='same',
                         strides=stride, activation=self.mid_activation)(out)
        out = Flatten()(out)
        out = concatenate([out, action_in])
        for l_size in self.fc_layer_sizes:
            out = Dense(l_size, activation=self.mid_activation)(out)
        out = Dense(self.action_dim, activation='linear')(out)
        model = Model([state_in, action_in], out)
        return model


def CNNQModel(in_shape, conv_layer_sizes, fc_layer_sizes, action_dim,
              kernel_sizes, strides, mid_activation, final_activation):
    state_in = Input(shape=in_shape)
    out = state_in
    for l_size, k_size, stride in zip(conv_layer_sizes, kernel_sizes, strides):
        out = Conv2D(filters=l_size, kernel_size=k_size, padding='same',
                     strides=stride, activation=mid_activation)(out)
    out = Flatten()(out)
    out = concatenate([out, action_in])
    for l_size in fc_layer_sizes:
        out = Dense(l_size, activation=mid_activation)(out)
    out = Dense(action_dim, activation='linear')(out)
    model = Model(state_in, out)
    return model
