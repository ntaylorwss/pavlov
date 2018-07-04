import numpy as np
import matplotlib.pyplot as plt
import JSAnimation.IPython_display as js_display
import IPython.display as ipy_display
import matplotlib.animation
from ..auxiliary.replay_buffer import ReplayBuffer
from ..auxiliary.monitor import Monitor
from .. import util


class Agent:
    """Composes an environment, data pipeline, model, and actor to perform reinforcement learning.

    The modular philosophy of the library means that this agent should be
    responsible for no more than calling its members to run timesteps,
    and to loop over timesteps to run episodes. The setup is:
        The environment provides a state,
        it's passed through the pipeline,
        the model makes a prediction for that state,
        the actor converts that prediction to an action,
        the environment consumes that action,
        the monitor keeps track of the important things,
        and sometimes the model learns from the replay buffer.

    Each of these components knows its Agent. This is how information is passed between modules.

    Parameters:
        env (gym.Env): the environment the agent is acting in.
        state_pipeline (list[functions]): a list of functions that the state
                                          is passed through sequentially.
        model (pavlov.Model): a reinforcement learning model that guides the agent.
                                options: DQNModel, DDPGModel.
        actor (pavlov.Actor): responsible for converting model predictions to actions.
        buffer_size (int): limit for how many observations to hold in replay buffer.
        batch_size (int): number of observations to pull from replay_buffer at fit time.
        warmup_length (int): number of random timesteps to execute before beginning
                             to learn and apply the model. Replay buffer will be populated.
        repeated_actions (int): number of env timesteps to repeat a chosen action for.
        report_freq (int): interval for printing to stdout, in number of episodes.
        state_dtype (type): numpy datatype for states to be stored in in replay buffer.

    Member variables:
        env (gym.Env): the environment the agent is acting in.
        env_state (np.array): current state of environment.
        state_pipeline (list[functions]): a list of functions that the state
                                          is passed through sequentially.
        model (pavlov.Model): a reinforcement learning model that guides the agent.
                                options: DQNModel, DDPGModel.
        actor (pavlov.Actor): responsible for converting model predictions to actions.
        replay_buffer (pavlov.ReplayBuffer): collection of historical observations.
        batch_size (int): number of observations to pull from replay_buffer at fit time.
        warmup_length (int): number of random timesteps to execute before beginning
                             to learn and apply the model. Replay buffer will be populated.
        repeated_actions (int): number of env timesteps to repeat a chosen action for.
        monitor (pavlov.Monitor): keeps track of metrics and logs them.
        renders_by_episode (list[np.array]): environment timestep renderings by episode.
    """
    # incompatible pairs of action space type and model type
    incompatibles = [('box', 'dqnmodel'), ('discrete', 'ddpgmodel'),
                     ('multidiscrete', 'ddpgmodel'), ('multibinary', 'ddpgmodel')]

    def __init__(self, env, state_pipeline, model, actor,
                 buffer_size, batch_size, warmup_length, repeated_actions=1,
                 report_freq=100, state_dtype=np.float32):
        # check if model and action space are compatible
        for space_type, model_type in self.incompatibles:
            if (env.action_space.__class__.__name__.lower() == space_type
                    and model.__class__.__name__.lower() == model_type):
                raise util.exceptions.ActionModelMismatchError(space_type, model_type)
        self.env = env
        self.env_state = self.env.reset()

        self.state_pipeline = state_pipeline
        self.replay_buffer = ReplayBuffer(buffer_size, state_dtype)
        self.model = model
        self.actor = actor
        self.monitor = Monitor(report_freq, '/var/log')
        self.state_pipeline.configure(self)
        self.replay_buffer.configure(self)
        self.model.configure(self)
        self.actor.configure(self)
        self.monitor.configure(self)

        self.batch_size = batch_size
        self.warmup_length = warmup_length
        self.repeated_actions = repeated_actions
        self.renders_by_episode = []

        self._empty_running_file()
        self._warmup_replay_buffer()

    def _empty_running_file(self):
        with open('/home/pavlov/agents/keep_running.txt', 'w'):
            pass

    def _warmup_replay_buffer(self):
        """Run replay buffer-populating timesteps before actually starting."""
        for i in range(self.warmup_length):
            reward, done = self.run_timestep(warming=True)
            if done:
                self.reset()

    def render_gif(self, episode=0):
        """Generates gif of agent's timesteps for given episode index."""
        frames = self.renders_by_episode[episode]
        plt.figure(figsize=(frames[0].shape[1] / 72.0,
                            frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = matplotlib.animation.FuncAnimation(plt.gcf(), animate,
                                                  frames=len(frames), interval=50)
        ipy_display(js_display.display_animation(anim, default_mode='loop'))

    def reset(self):
        """Resets environment to initial state for new episode."""
        self.env_state = self.env.reset()

    def run_timestep(self, warming=False, render=False):
        """Run one timestep in the environment.

        - Process current state with `state_pipeline`.
        - Generate prediction from `model`.
        - Convert prediction to action twice; one format for replay buffer, one for environment.
        - Apply action in environment, observe reward and next state.
        - Store experience in replay buffer.
        - Every `batch_size` timesteps, fit model to random batch from replay buffer.
        """
        start_state = self.state_pipeline.transform(self.env_state)
        if warming:
            action_for_model, action_for_env = self.actor.warming_action()
        else:
            pred = self.model.predict(start_state)
            action_for_model, action_for_env = self.actor.convert_pred(pred)  # includes exploration

        timestep_reward = 0.
        for i in range(self.repeated_actions):
            if not warming and render:
                rendered_frame = self.env.render(mode='rgb_array')
                self.renders_by_episode[-1].append(rendered_frame)
            self.env_state, frame_reward, done, info = self.env.step(action_for_env)
            timestep_reward += frame_reward
            if done:
                break

        if not warming:
            self.actor.step(done)
            self.monitor.step(timestep_reward)

        if done:
            next_state = np.zeros(self.replay_buffer['states'].shape[1:])
        else:
            next_state = self.state_pipeline.transform(self.env_state)

        self.replay_buffer.add(start_state, action_for_model, timestep_reward, next_state, done)
        if not warming and self.replay_buffer.current_size >= self.batch_size:
            batch = self.replay_buffer.get_batch(self.batch_size)
            self.model.fit(**batch)

        return timestep_reward, done

    def run_episode(self, render=False, log=True):
        """Apply `run_timestep` until episode terminates. Apply monitor for logging."""
        self.renders_by_episode.append([])
        total_reward = 0.
        while True:
            reward, done = self.run_timestep(render=render)
            total_reward += reward
            if done:
                self.monitor.write_summary()
                self.monitor.new_episode()
                self.reset()
                break
        if log:
            self.monitor.log_to_stdout()

    def run_indefinitely(self, render=False, log=True):
        """Apply run_episode continuously until keep_running.txt is populated with 'False'."""
        while True:
            self.run_episode(render=render, log=log)
            with open('/home/pavlov/agents/keep_running.txt') as f:
                contents = f.read().strip()
                if contents == 'False':
                    print("Stopping.")
                    break
