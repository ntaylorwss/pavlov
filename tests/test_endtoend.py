import sys
sys.path.append('/home')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pavlov
import gym
import tensorflow as tf


class TestPavlov:
    def test_basic_discrete_agent(self):
        # arrange
        env = gym.make("CartPole-v0")
        base_config = {
            'layer_sizes': [128],
            'activation': 'relu'
        }
        dqn_config = {
            'gamma': 0.99,
            'tau': 1.0,
            'optimizer': tf.keras.optimizers.SGD(lr=0.03)
        }
        topology = pavlov.models.topology.DenseTopology(**base_config)
        model = pavlov.models.DQNModel(topology, **dqn_config)

        epsilon_schedule = pavlov.auxiliary.schedules.LinearDecaySchedule(1.0, 0.1, 500, -1)
        actor = pavlov.actors.EpsilonGreedyActor(epsilon_schedule)
        buffer_size = 10000
        batch_size = 2

        pline = pavlov.pipeline.Pipeline()
        agent = pavlov.agents.Agent(env,
                                    state_pipeline=pline,
                                    model=model, actor=actor,
                                    buffer_size=buffer_size, batch_size=batch_size,
                                    report_frequency=10, warmup_length=50)


        # act
        agent.run_episode(render=False, do_logging=True)
        print("Done")

    def test_intermediate_discrete_env(self):
        env = gym.make('Breakout-v0')
        topology_config = {
            'layer_sizes': [128],
            'activation': 'relu'
        }
        topology = pavlov.models.topology.DenseTopology(**topology_config)

        dqn_config = {
            'gamma': 0.99,
            'tau': 1.0,
            'optimizer': tf.keras.optimizers.Adam(0.0001)
        }
        model = pavlov.models.DQNModel(topology, **dqn_config)

        epsilon_schedule = pavlov.auxiliary.schedules.LinearDecaySchedule(1.0, 0.1, 500, -1)
        actor = pavlov.actors.EpsilonGreedyActor(epsilon_schedule)
        buffer_size = 10000
        batch_size = 64

        pline = pavlov.pipeline.Pipeline()
        pline.add(pavlov.transformations.rgb_to_grey())
        pline.add(pavlov.transformations.downsample(new_shape=(84, 84)))
        pline.add(pavlov.transformations.combine_consecutive(2, 'max'))
        pline.add(pavlov.transformations.stack_consecutive(4))

        agent = pavlov.agents.Agent(env,
                                    state_pipeline=pline,
                                    model=model, actor=actor,
                                    buffer_size=buffer_size, batch_size=batch_size,
                                    report_frequency=1, warmup_length=5)
        agent.run_episode(render=False, do_logging=True)
        print("Done")

    def test_basic_continuous_env(self):
        env = gym.make('MountainCarContinuous-v0')
        base_config = {
            'layer_sizes': [128],
            'activation': 'relu'
        }
        ddpg_config = {
            'actor_activation': 'softmax',
            'gamma': 0.99,
            'tau': 0.1,
            'actor_optimizer': tf.train.AdamOptimizer(0.0001),
            'critic_optimizer': tf.keras.optimizers.Adam(lr=0.0001)
        }
        topology = pavlov.models.topology.DenseTopology(**base_config)
        model = pavlov.models.DDPGModel(topology, **ddpg_config)

        epsilon_schedule = pavlov.auxiliary.schedules.LinearDecaySchedule(1.0, 0.1, 500, -1)
        actor = pavlov.actors.EpsilonGreedyActor(epsilon_schedule)
        buffer_size = 10000
        batch_size = 64

        pline = pavlov.pipeline.Pipeline()
        agent = pavlov.agents.Agent(env,
                                    state_pipeline=pline,
                                    model=model, actor=actor,
                                    buffer_size=buffer_size, batch_size=batch_size,
                                    report_frequency=1, warmup_length=50)
        agent.run_episode(render=False, do_logging=True)
        print("Done")
