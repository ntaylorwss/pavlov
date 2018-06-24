from keras.layers import Input, Dense, Conv2D, Flatten, concatenate, Activation


def cnn_model_base(in_shape, conv_layer_sizes, fc_layer_sizes,
                   kernel_sizes, strides, activation):
    state_in = Input(shape=in_shape)
    out = state_in
    for l_size, k_size, stride in zip(conv_layer_sizes, kernel_sizes, strides):
        out = Conv2D(filters=l_size, kernel_size=k_size, padding='same',
                     strides=stride, activation=activation)(out)
    out = Flatten()(out)
    for l_size in fc_layer_sizes:
        out = Dense(l_size, activation=activation)(out)
    return state_in, out


def dense_model_base(in_shape, layer_sizes, activation):
    state_in = Input(shape=in_shape)
    out = state_in
    for l_size in layer_sizes:
        out = Dense(l_size, activation=activation)(out)
    return state_in, out
