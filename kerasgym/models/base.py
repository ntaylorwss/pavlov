from keras.layers import Input, Dense, Conv2D, Flatten, concatenate, Activation


def cnn_model_base(in_shape, conv_layer_sizes, fc_layer_sizes,
                   kernel_sizes, strides, activation):
    """Creates headless Keras computation graph for a convolutional architecture.

    Starts with convolutional layers according to `conv_layer_sizes`, `kernel_sizes`, and `strides`.
    Flattens those, then passes flattened vector to dense layers according to `fc_layer_sizes`.
    Uses `activation` as activation along the way.
    Always uses `same` padding.

    Parameters:
        in_shape (tuple(int)): dimensions/shape of input state, not including observation dimension.
        conv_layer_sizes (list[int]): number of filters in each convolutional layer.
        fc_layer_sizes (list[int]): number of units in each dense layer.
        kernel_sizes (list[2-tuple(int)]): 2-dimensional size of convolutional kernel.
        strides (list[2-tuple(int)]): dimensions of stride in convolution.
        activation (str): activation function to be used in each layer.
    """
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
    """Creates headless Keras computation graph for a fully connected architecture.

    Simply passes input through a series of dense layers, with chosen `activation`.

    Parameters:
        in_shape (tuple(int)): dimensions/shape of input state, not including observation dimension.
        layer_sizes (list[int]): number of units in each dense layer.
        activation (str): activation function to be used in each layer.
    """
    state_in = Input(shape=in_shape)
    out = state_in
    for l_size in layer_sizes:
        out = Dense(l_size, activation=activation)(out)
    return state_in, out
