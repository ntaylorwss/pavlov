# KerasGym: A Modular, Composable Approach to Reinforcement Learning

KerasGym is an approach to reinforcement learning focused on a modular design. This design allows the building of reinforcement learning agents by selecting from among a set of options for each of the following components:

- *Environment*. Any environment that complies with the OpenAI Gym interface can be used. Note that environments that do not comply with this interface cannot be used. For an example of how to register your custom environment as a Gym environment, see [Custom Environments](#custom-environments).
- *State pipeline*. A pipeline is a sequence of pure functions that transform their input in some way. A state pipeline in KerasGym is defined as a list of functions. There is a collection of common functions in the module `kerasgym.pipeline`, but you can easily write your own function and place it in the list. See [State Pipeline](#state-pipeline) for details.
- *Model*. The model is the heart of reinforcement learning; this is where the actual learning takes place. For a full list of available models (being updated continuously), see [Models](#models).
- *Actor*. The actor is responsible for converting the prediction of the Model into an action to be consumed by the Environment. This includes both exploration and conversion to the correct format for the environment's action space. The Actor will automatically and silently detect the type of action required by the environment (whether continuous, discrete, multi-dimensional discrete, multi-dimensional binary), and perform conversion according to the kind of model and action space it's working with. For more information on the kinds of exploration policies that are available, see [Exploration](#exploration).

## Installation

## Getting Started
Here's an example of an end-to-end usage of KerasGym to produce an agent for Breakout and have it run forever. Note that this agent doesn't actually solve Breakout with these settings, but once I figure out what settings solve it I will update the example.

```python
import gym
from keras import optimizers

from kerasgym import pipeline
from kerasgym import models
from kerasgym import exploration
from kerasgym import agents

env = gym.make('Breakout-v0')

topology_config = {
    'layer_sizes': [128],
    'activation': 'relu'
}
topology = models.topology.DenseTopology(**topology_config)

dqn_config = {
    'gamma': 0.99,
    'tau': 1.0,
    'optimizer': optimizers.Adam(0.0001)
}
model = models.DQNModel(topology, **dqn_config)

epsilon_schedule = auxiliary.schedules.LinearDecay(1.0, 0.1, 500, -1)
actor = actors.EpsilonGreedyActor(epsilon_schedule)
buffer_size = 10000
batch_size = 64

pipeline = pipeline.Pipeline()
pipeline.add(transformations.rgb_to_grey())
pipeline.add(transformations.downsample(new_shape=(84, 84)))
pipeline.add(transformations.combine_consecutive(2, 'max'))
pipeline.add(transformations.stack_consecutive(4))

agent = agents.Agent(env,
                     state_pipeline=pipeline,
                     model=model, actor=actor,
                     buffer_size=buffer_size, batch_size=batch_size,
                     report_freq=1, warmup_length=50)
agent.run_indefinitely()
```

## Environments
### Gym Environments
KerasGym is equipped to function with any environment that follows the [OpenAI Gym API](https://github.com/openai/gym/blob/master/gym/core.py#L11). For environments that are native to Gym, usage is straightforward. Simply use `gym.make` to generate an environment, and pass it to the Agent, as shown in [Getting Started](#getting-started).

### Custom Environments
A custom environment can surely be used with KerasGym, as long as it is first made to comply with the Gym API, and is properly registered as a Gym environment. What follows is all that is necessary to create a custom environment that complies with the OpenAI Gym API.

A valid custom environment class must be necessarily equipped with the following _member variables_:

- `self.observation_space`: An instance of a class from the `gym.spaces` module. Details are below under [Spaces](#spaces).
- `self.action_space`: An instance of a class from the `gym.spaces` module. Details are below under [Spaces](#spaces).

And it must be necessarily equipped with the following _methods_:

- `self.seed(seed)`: It's not necessary for this to actually do anything if it's not needed, but it must be part of the class. A simple `return gym.utils.seeding.np_random(seed)[1:]` in the body of the function will work.
- `self.render(mode, close)`: Again, not necessary for this to actually do anything if it doesn't make sense for your environment to render. In this event, you should use `return []` in the body as a dummy.
- `self.reset()`: This is a key method, as it defines what happens at the termination of an episode. This should contain all your logic to reset the environment to its initial state.
- `self.step(action)`: This is the main method, which defines a single timestep of the environment. It consumes an action and transitions the environment based on it.

See Gym documentation for more details on what these methods are for and what they're expected to return.

From there, here is a snippet of slightly abstracted code (in the sense of the variable names being abstracted) for registering this environment with Gym:

```python
from gym.envs.registration import register
from mymodule.mysubmodule import MyCustomEnvironment

def register_custom_env(env_init_arg_1, env_init_arg_n, env_name='MyCustomEnvironment'):
    kwargs = {'env_init_arg_1': env_init_arg_1, 'env_init_arg_n': env_init_arg_n}
    register(id=env_name, entry_point='mymodule.mysubmodule:MyCustomEnvironment',
             kwargs=kwargs)
    return gym.make(env_name)
```

Note that an environment can only be registered once per Python session.

### Spaces
Currently, Gym supports 4 types spaces for both observations and actions, found in the module `gym.spaces`: they are `Box`, `Discrete`, `MultiDiscrete`, and `MultiBinary`. All of these spaces are supported by KerasGym; though for Box, actions can only be up to 2D, while for observations, they can be any n-dimensional shape.

Keep in mind when writing your environment that the output of any Space, including both the observation space and action space, must be the same at every timestep.

## State Pipeline
Often, a raw state is not going to be appropriate for effective learning, and transformations are required. The fundamental philosophy of this library is that data processing and feature engineering follows a "pipeline" structure. This means that a series of transformations are applied sequentially to each input state, to produce the final state formatting. These transformations should be implemented as individual functions. At that point, a list of these functions can be passed to the Agent. An example of a pipeline would be:

```python
from kerasgym import pipeline
pipeline = [pipeline.rgb_to_grey(method='luminosity'), # convert 3D RGB to 2D greyscale
            pipeline.downsample(new_shape=(84, 84)), # resize image to smaller size by interpolation
            pipeline.combine_consecutive(n_states=2, fun='max'), # each state is the max of current and previous states
            pipeline.stack_consecutive(n_states=4) # make each state a time-series of the last 4 states
]
```

The pipelined nature of this setup means that `combine_consecutive` and `stack_consecutive`, for example, can cooperate smoothly; `stack_consecutive` will make a stack of 4 combined states, since that's the output of the `combine_consecutive` step. The result of this whole pipeline is a stack of 4 84x84 greyscale max-outs over 2 consecutive frames, at each timestep.

Note that the pipeline functions, such as `pipeline.rgb_to_grey()`, themselves return functions. `pipeline` is a list of functions. This design allows "hyperparameters" of functions, such as the shape of downsampling, to be easily specified.

### Custom Functions
The only requirement for a pipeline function is that its signature takes `state` and `env` as arguments. Many pipeline functions will not make use of `env` information, but some will, so it's a necessary consistency. The recommendation is also to write pipeline functions as functors; that is, functions that return functions. As mentioned above, this allows easy specification and tuning of pipeline functions, and a readable syntax. Here's an example:

```python
def reshape_array(new_shape):
    def _reshape_array(state, env):
        return np.reshape(state, new_shape)
    return _reshape_array
```

The convention to start the inner function with an underscore and name it identically to the outer function is just personal preference.

## Models
At the moment, there are 2 reinforcement learning algorithms provided by KerasGym:

- Deep Q Network (DQN)
    - Double DQN is implemented, which is to say there is a target network; setting the parameter `tau` to `1.0` will effectively negate the target network and make it plain DQN
- Deep Deterministic Policy Gradient (DDPG)

The parameters for each model are specified in the documentation and docstrings.

One core concept of KerasGym is that the feature extraction component of a neural network should be separated from the action selection component; for this reason, all classes for RL algorithms will expect to be given a headless Keras computation graph, or `kerasgym.models.Topology`, as input.

### Custom Model Configuration
To write an entirely customized Keras model graph, do the following:

1. Start your new model class (let's call it `MyModel`), and have it inherit the `kerasgym.models.topology.Topology` base class:
    - e.g. `class MyModel(kerasgym.models.topology.Topology):`
2. Override the method `define_graph(self, ...)`, filling in `...` with whatever parameters your configuration requires, e.g. `dense_layer_sizes`. Alternatively, you can have no parameters:
    - e.g. `def define_graph(self, dense_layer_sizes, activation, weight_initializer):`
3. Initialize a new instance of your model class by passing your arguments for the parameters you provided in the declaration of `define_graph`:
    - e.g. `model = MyModel(dense_layer_sizes=[16, 8, 4], activation='relu', weight_initializer='glorot_normal')`

With this design, you can create any Keras graph you want, and parameterize it however you wish.

### Input Layer
A key note about this design is that the Input layer of the model is taken care of by the Agent internally. This layer will have the correct shape according to the output of your state pipeline, and it can be accessed through `self.input` in your child class of `Topology`. As an abbreviated example:

```python
class MyCustomModel(Topology):
    def define_graph(self, layer1_size, ...):
        X = Dense(layer1_size)(self.input) # accessing the Input layer through self.input
        ...
```

This way you do not have to care about making shapes agree whatsoever.

## Exploration
At the moment, there is 1 method of exploration provided by KerasGym:

- Epsilon Greedy

For any exploration module that requires an important value, such as the epsilon in epsilon greedy, there is a requirement to provide a schedule for that value's progression throughout training. Every schedule has a starting value, and will typically have an ending value, as well as other particular parameters. The schedules currently available are:

- Constant. The value never changes.
- Linear decay. The value decays linearly towards some point over some number of steps, and then flatlines.
- Exponential decay. The value decays exponentially towards some point over some number of steps, and then asymptotes.
- Scoping periodic. The value follows an exponential decay over some number of steps, but also oscillates in a sinusoidal wave, with specified period and amplitude. The result is a sinusoidal wave that eventually decays and flatlines.
