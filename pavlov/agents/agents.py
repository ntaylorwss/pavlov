"""Agents compose an environment, state pipeline, model, and actor to do reinforcement learning.

Agents have a simple API: you can reset the environment, run a timestep, run an episode,
or run the agent indefinitely (and stop safely whenever you want).
"""

import datetime
import numpy as np
import cv2
from custom_inherit import DocInheritMeta
from ..auxiliary.replay_buffer import ReplayBuffer
from ..auxiliary.monitor import Monitor
from .. import util


class Agent(metaclass=DocInheritMeta(style="numpy")):
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

    Parameters
    ----------
    env : gym.Env
        the environment the agent is acting in.
    state_pipeline : list of functions
        a list of functions that the state is passed through sequentially.
    model : pavlov.Model
        a reinforcement learning model that guides the agent.
    actor : pavlov.Actor
        responsible for converting model predictions to actions.
    buffer_size : int
        limit for how many observations to hold in replay buffer.
    batch_size : int
        number of observations to pull from replay_buffer at fit time.
    warmup_length : int
        number of random timesteps to execute before beginning
        to learn and apply the model. Replay buffer will be populated.
    repeated_actions : int
        number of env timesteps to repeat a chosen action for.
    report_frequency : int
        interval for printing to stdout, in number of episodes.
    state_dtype : type
        numpy datatype for states to be stored in in replay buffer.

    Attributes
    ----------
    env : gym.Env
        the environment the agent is acting in.
    env_state : np.array
        current state of environment.
    state_pipeline : list of functions
        a list of functions that the state is passed through sequentially.
    model : pavlov.Model
        a reinforcement learning model that guides the agent.
    actor : pavlov.Actor
        responsible for converting model predictions to actions.
    replay_buffer : pavlov.ReplayBuffer
        collection of historical observations.
    batch_size : int
        number of observations to pull from replay_buffer at fit time.
    warmup_length : int
        number of random timesteps to execute before beginning
        to learn and apply the model. Replay buffer will be populated.
    repeated_actions : int
        number of env timesteps to repeat a chosen action for.
    monitor : pavlov.Monitor
        keeps track of metrics and logs them.
    renders_by_episode : list of np.array
        environment timestep renderings by episode.
    """
    # incompatible pairs of action space type and model type
    incompatibles = [('box', 'dqnmodel'), ('discrete', 'ddpgmodel'),
                     ('multidiscrete', 'ddpgmodel'), ('multibinary', 'ddpgmodel')]

    def __init__(self, env, state_pipeline, model, actor,
                 buffer_size, batch_size, warmup_length, repeated_actions=1,
                 report_frequency=100, state_dtype=np.float32):
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
        self.monitor = Monitor(report_frequency, '/var/log')
        self.state_pipeline.configure(self)
        self.replay_buffer.configure(self)
        self.model.configure(self)
        self.actor.configure(self)
        self.monitor.configure(self)

        self.batch_size = batch_size
        self.warmup_length = warmup_length
        self.repeated_actions = repeated_actions
        self.renders_by_episode = [[]]
        self.start_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%s')

        self._warmup_replay_buffer()

    def _warmup_replay_buffer(self):
        """Run replay buffer-populating timesteps before actually starting."""
        for i in range(self.warmup_length):
            done = self.run_timestep(warming=True)
            if done:
                self.reset()

    def episode_to_mp4(self, episode_num, out_dir):
        """Generates mp4 of agent's timesteps for given episode number.

        Only works with environments that render images at each timestep.

        Parameters
        ----------
        episode_num : int
            episode number that you want to take a video of (one-indexed).
        out_dir : str
            directory where you would like to place video file.
            filename is auto-generated.
        """
        frames = self.renders_by_episode[episode-1]
        shape = frames[0].shape
        if not ((len(shape) == 2) or (len(shape) == 3 and shape[2] == 3)):
            raise TypeError("Environment renderings are not images")

        size = frames[0].shape[:-1]
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_out = cv2.VideoWriter()
        filename = '{}/{}-episode{}.mp4'.format(self.start_timestamp, out_dir, episode)
        vid_out.open(filename, fourcc, fps, size, True)
        for frame in frames:
            vid_out.write(frame)
        vid_out.release()

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

        Parameters
        ----------
        warming : bool
            indicate whether this is a step meant to simply populate the replay buffer.
        render : bool
            indicate whether to log an image of the environment for video generation.
        """
        start_state = self.state_pipeline.transform(self.env_state)
        if warming:
            action_for_model, action_for_env = self.actor.warming_action()
        else:
            pred = self.model.predict(start_state)
            action_for_model, action_for_env = self.actor.convert_pred(pred)  # includes exploration

        timestep_reward = 0.
        for i in range(self.repeated_actions):
            if render:
                if not warming:
                    rendered_frame = self.env.render(mode='rgb_array')
                    self.renders_by_episode[-1].append(rendered_frame)
            else:
                self.renders_by_episode[-1].append([])
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

        return done

    def run_episode(self, render=False, do_logging=True):
        """Apply `run_timestep` until episode terminates.

        Parameters
        ----------
        render : bool
            indicate whether to log an image of the environment for video generation.
        do_logging : bool
            indicate whether to print out metrics for episode.
        """
        self.renders_by_episode.append([])
        while True:
            done = self.run_timestep(render=render)
            if done:
                self.monitor.new_episode(do_logging)
                self.reset()
                break

    def run_indefinitely(self, render=False, log=True):
        """Apply run_episode in an infinite loop; terminated by KeyboardInterrupt.

        The keyboard interrupt will be handled by first finishing the running episode,
        then terminating the loop and thus function.

        Parameters
        ----------
        render : bool
            indicate whether to log an image of the environment for video generation.
        do_logging : bool
            indicate whether to print out metrics for episode.
        """
        with util.interrupt.AtomicLoop() as loop:
            while loop.run:
                self.run_episode(render=render, do_logging=log)
