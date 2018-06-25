import numpy as np
import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython import display
from .replay_buffer import ReplayBuffer
from .metrics import Monitor
from ..util import get_action_type, ActionModelMismatchError


class Agent:
    # incompatible pairs of action space type and model pred type
    incompatibles = [('box', 'value')]

    def __init__(self, env, state_pipeline, model, actor,
                 buffer_size, batch_size, warmup_length, repeated_actions=1,
                 report_freq=100, state_dtype=np.float32):
        self.env = env
        self.env_state = self.env.reset()
        self.state_pipeline = state_pipeline
        self.model = model
        # check if model and action space are compatible
        for space_type, pred_type in self.incompatibles:
            if (get_action_type(self.env.action_space) == space_type
                    and self.model.pred_type == pred_type):
                raise ActionModelMismatchError(space_type, pred_type)
        self.model.configure(self.env.action_space)
        self.actor = actor
        self.actor.configure(self)
        self.replay_buffer = ReplayBuffer(buffer_size, model.in_shape,
                                          self.env.action_space, state_dtype)
        self.batch_size = batch_size
        self.warmup_length = warmup_length
        self.repeated_actions = repeated_actions
        self.monitor = Monitor(self, '/home/kerasgym/logdir', n_episode_avg=10,
                               report_freq=report_freq)
        # display
        self.renders_by_episode = []
        # empty keep_running file
        with open('/home/kerasgym/agents/keep_running.txt', 'w'):
            pass
        # warm up agent by running some random episodes and building a memory
        self._warmup()

    def render_gif(self, episode=0):
        frames = self.renders_by_episode[episode]
        plt.figure(figsize=(frames[0].shape[1] / 72.0,
                            frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        display.display(display_animation(anim, default_mode='loop'))

    def reset(self):
        self.env_state = self.env.reset()

    def _transform_state(self, state):
        out = state.copy()
        for transformation in self.state_pipeline:
            out = transformation(out, self.env)
        return out

    def _warmup(self):
        for i in range(self.warmup_length):
            reward, done = self.run_timestep(warming=True)
            if done:
                self.reset()

    def run_timestep(self, warming=False, render=False):
        start_state = self._transform_state(self.env_state)
        if warming:
            a_for_model, a_for_env = self.actor.warming_action()
        else:
            pred = self.model.predict(start_state)
            a_for_model, a_for_env = self.actor.convert_pred(pred)  # includes exploration

        reward = 0.
        for i in range(self.repeated_actions):
            if not warming and render:
                self.renders_by_episode[-1].append(self.env.render(mode='rgb_array'))
            self.env_state, r, done, info = self.env.step(a_for_env)
            reward += r
            if done:
                break

        if not warming:
            self.actor.step(done)
            self.monitor.step(reward)

        if done:
            next_state = np.zeros(self.replay_buffer.states.shape[1:])
        else:
            next_state = self._transform_state(self.env_state)

        self.replay_buffer.add(start_state, a_for_model, reward, next_state, done)
        if not warming and self.replay_buffer.current_size >= self.batch_size:
            batch = self.replay_buffer.get_batch(self.batch_size)
            self.model.fit(**batch)

        return reward, done

    def run_episode(self, render=False):
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
            self.monitor.log_to_stdout()

    def run_indefinitely(self, render=False):
        while True:
            self.run_episode(render=render)
            with open('/home/kerasgym/agents/keep_running.txt') as f:
                contents = f.read().strip()
                if contents == 'False':
                    print("Stopping.")
                    break
