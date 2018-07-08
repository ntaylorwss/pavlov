import sys
sys.path.append('/home')

import pavlov
import gym
import keras.optimizers

class TestPavlov:
    def setup(self):
        self.basic_discrete_env = gym.make("CartPole-v0")
        self.intermediate_discrete_env = gym.make("Breakout-v0")

    def test_basic_discrete_agent(self):
        # arrange
        base_config = {
            'layer_sizes': [128],
            'activation': 'relu'
        }
        dqn_config = {
            'gamma': 0.99,
            'tau': 1.0,
            'optimizer': keras.optimizers.SGD(lr=0.03)
        }
        topology = pavlov.models.topology.DenseTopology(**base_config)
        model = pavlov.models.DQNModel(topology, **dqn_config)

        epsilon_schedule = pavlov.auxiliary.schedules.LinearDecaySchedule(1.0, 0.1, 500, -1)
        actor = pavlov.actors.EpsilonGreedyActor(epsilon_schedule)
        buffer_size = 10000
        batch_size = 64

        pline = pavlov.pipeline.Pipeline()
        agent = pavlov.agents.Agent(self.basic_discrete_env,
                                    state_pipeline=pline,
                                    model=model, actor=actor,
                                    buffer_size=buffer_size, batch_size=batch_size,
                                    report_frequency=10, warmup_length=50)

        # act
        agent.run_episode(render=False, do_logging=True)

"""
    def test_intermediate_discrete_env(self):
        topology_config = {
            'layer_sizes': [128],
            'activation': 'relu'
        }
        topology = pavlov.models.topology.DenseTopology(**topology_config)

        dqn_config = {
            'gamma': 0.99,
            'tau': 1.0,
            'optimizer': keras.optimizers.Adam(0.0001)
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

        agent = pavlov.agents.Agent(self.intermediate_discrete_env,
                                    state_pipeline=pline,
                                    model=model, actor=actor,
                                    buffer_size=buffer_size, batch_size=batch_size,
                                    report_frequency=1, warmup_length=5)
        agent.run_episode(render=False, do_logging=True)
"""
