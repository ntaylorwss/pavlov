from keras.layers import Input, Flatten, concatenate, Activation
from keras.layers import Dense, Conv2D


class Topology:
    """Base class for creating headless Keras computation graphs with arbitrary architecture.

    Input layer is pre-defined, and resides in `self.input`,
    to be used by user-defined child method.
    Child init methods are not required.

    Any parameters required by the graph definition, e.g. layer sizes, should be made
    parameters for the `define_graph` method. They will become the initialization
    parameters of the class.
    For example, if you wanted `layer_sizes, activation` to be the parameters required
    to define the model, you would make those the parameters of the
    child method `define_graph`, and then initialize
    the object with `Topology(layer_sizes, activation)`.

    Parameters
    ----------
    **define_graph_kwargs
        kwargs to be passed to define_graph method.
    """
    def __init__(self, **define_graph_kwargs):
        """Store arguments for parameters required by define_graph in a dictionary.
        Note: args must be passed to init as kwargs, not positional args."""
        # Part 1: the keywords and args passed are composed into a dictionary and stored
        self.define_graph_kwargs = define_graph_kwargs

    def configure(self, agent):
        """Set up input and output layers and store them in member variables.

        Input is configured according to the associated agent.
        Output is the return value of define_graph, as defined by the child class.

        Parameters
        ----------
        agent : pavlov.Agent
            Agent to associate with model.
        """
        self.input = Input(shape=agent.state_pipeline.out_dims,
                           dtype=agent.replay_buffer.state_dtype)
        # Part 2: the dictionary of args is unpacked and passed to define_graph
        self.output = self.define_graph(**self.define_graph_kwargs)

    def define_graph(self, **define_graph_kwargs):
        """Define any arbitrary Keras graph, making use of the kwargs passed to init.

        Parameters
        ----------
        **define_graph_kwargs
            Can be any arguments necessary to define a model.

        Raises
        ------
        NotImplementedError.
        """
        # Part 3: define_graph is necessarily declared with the right keyword args
        raise NotImplementedError


class CNNTopology(Topology):
    """Creates headless Keras computation graph for a convolutional architecture.

    Starts with convolutional layers according to `conv_layer_sizes`, `kernel_sizes`, and `strides`.
    Flattens those, then passes flattened vector to dense layers according to `fc_layer_sizes`.
    Uses `activation` as activation along the way.
    Always uses `same` padding.
    The input shape is discovered at configuration time, from the state pipeline.
    """
    def define_graph(self, conv_layer_sizes, fc_layer_sizes, kernel_sizes, strides, activation):
        """Defines headless Keras graph for a CNN.

        Parameters
        ----------
        conv_layer_sizes : list of int
            number of filters in each convolutional layer.
        fc_layer_sizes : list of int
            number of units in each dense layer.
        kernel_sizes : list of 2-tuple of int
            2-dimensional size of convolutional kernel.
        strides : list of 2-tuple of int
            dimensions of stride in convolution.
        activation : str
            activation function to be used in each layer.

        Returns
        -------
        out : keras.Layer
            Final Keras layer of headless architecture, to be passed on to the final layers.
        """
        out = self.input
        for l_size, k_size, stride in zip(conv_layer_sizes, kernel_sizes, strides):
            out = Conv2D(filters=l_size, kernel_size=k_size, padding='same',
                         strides=stride, activation=activation)(out)
        out = Flatten()(out)
        for l_size in fc_layer_sizes:
            out = Dense(l_size, activation=activation)(out)
        return out


class DenseTopology(Topology):
    """Creates headless Keras computation graph for a fully connected architecture.

    Passes input through a series of dense layers, with chosen `activation`.
    """
    def define_graph(self, layer_sizes, activation):
        """Defines headless Keras graph for a fully connected network.

        Parameters
        ----------
        layer_sizes : list of int
            number of units in each dense layer.
        activation : str
            activation function to be used in each layer.

        Returns
        -------
        out : keras.Layer
            Final Keras layer of headless architecture, to be passed on to the final layers.
        """
        out = self.input
        for L in layer_sizes:
            out = Dense(L, activation=activation)(out)
        return out
