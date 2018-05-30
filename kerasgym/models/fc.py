import numpy as np
from keras.models import Model
from keras.layers import Input, Dense

def headless_fc(in_shape, layer_sizes, mid_activation):
    def _headless_fc():
        in_layer = Input(in_shape)
        out = in_layer
        for s in layer_sizes:
            out = Dense(s, activation=mid_activation)(out)
        return in_layer, out
    return _headless_fc
