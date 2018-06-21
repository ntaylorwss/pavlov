import numpy as np
from collections import deque
import time
from datetime import datetime
import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython import display
from .replay_buffer import ReplayBuffer
from .metrics import Monitor


class Agent:
    def __init__(self, env, state_pipeline, model, explorer,
                 buffer_size, batch_size, warmup_length, repeated_actions=1,
                 report_freq=100, state_dtype=np.float32):
        self.env = env
        self.env_state = self.env.reset()
        self.state_pipeline = state_pipeline
        self.model = model
        self.explorer = explorer
        self.explorer.configure(self)
        self.replay_buffer = ReplayBuffer(buffer_size, model.in_shape,
                                          model.action_dim, state_dtype)
        self.batch_size = batch_size
        self.warmup_length = warmup_length
        self.repeated_actions = repeated_actions
        self.monitor = Monitor(self, '/home/kerasgym/logdir', n_episode_avg=10, report_freq=report_freq)
        # display
        self.renders_by_episode = []
        # empty keep_running file
        with open('/home/kerasgym/agents/keep_running.txt', 'w') as f:
            pass
        # warm up agent by running some random episodes and building a memory
        self._warmup()

    def render_gif(self, episode=0):
        frames = self.renders_by_episode[episode]
        plt.figure(figsize=(frames[0].shape[1] / 72.0,
                                 frames[0].shape[0] / 72.0), dpi = 72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        display.display(display_animation(anim, default_mode='loop'))

    def reset(self):
        self.env_state = self.env.reset()

    def _warmup(self):
        for i in range(self.warmup_length):
            reward, done = self.run_timestep(warming=True)
            if done:
                self.reset()

    def run_timestep(self, warming=False, render=False):
        start_state = self.env_state
        for transformation in self.state_pipeline:
            start_state = transformation(start_state, self.env)

        if warming:
            consumable_action = self.env.action_space.sample()
        else:
            pred = self.model.predict(start_state)
            action = self.explorer.explore_and_convert(pred)



    def run_episode(self, render=False):
        self.renders_by_episode.append([])
        total_reward = 0.
        while True:
            reward, done = self.run_timestep(render=render)
            total_reward += reward
            if done:
                self.monitor.write_summary()
                self.monitor.new_episode()
                break
            self.reset()
            self.monitor.log()

    def run_indefinitely(self, render=False):
        while True:
            self.run_episode(render=render)
            with open('/home/kerasgym/agents/keep_running.txt') as f:
                contents = f.read().strip()
                if contents == 'False':
                    print("Stopping.")
                    break

class DQNAgent(Agent):
    def __init__(self, env, state_processing_fns, explorer,
                 buffer_size, batch_size, warmup_length, repeated_actions=1,
                 report_freq=100, state_dtype=np.float32):
        super().__init__(env, state_processing_fns, batch_size, warmup_length,
                         repeated_actions, report_freq)
        self.model = model
        self.explorer = explorer
        self.explorer.add_to_agent(self)
        keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.replay_buffer = ReplayBuffer(buffer_size, model.in_shape,
                                          model.action_dim, state_dtype, keys)

    def run_timestep(self, warming=False, render=False):
        start_state = self.env_state
        for state_processor in self.state_processors:
            start_state = state_processor(start_state, self.env)

        if warming:
            # CONTINUE HERE













class Agent:
    """Defines interface of agents, implements run_episode and reset. run_timestep should be
       defined by the child class."""

    def __init__(self, env, state_processing_fns, model, ptoc_fn, ctol_fn, explorer,
                 buffer_size, batch_size, warmup_length=0, repeated_actions=1,
                 report_freq=100, state_dtype=np.float32):
        self.env = env
        self.reset()
        self.state_processors = state_processing_fns
        self.model = model
        self.pred_to_consumable = ptoc_fn # for passing to environment
        self.consumable_to_learnable = ctol_fn # for storing and learning
        self.explorer = explorer
        self.explorer.add_to_agent(self)
        self.replay_buffer = ReplayBuffer(buffer_size, model.in_shape, model.action_dim, dtype=state_dtype)
        self.batch_size = batch_size
        self.warmup_length = warmup_length
        self.repeated_actions = repeated_actions
        self.monitor = Monitor(self, '/home/kerasgym/logdir', n_episode_avg=10)

        # display
        self.renders_by_episode = []

        # empty keep_running file
        with open('/home/kerasgym/agents/keep_running.txt', 'w') as f:
            pass

        # warm up agent by running some random episodes and building a memory
        self._warmup()

    def reset(self):
        self.env_state = self.env.reset()

    def _warmup(self):
        for i in range(self.warmup_length):
            reward, done = self.run_timestep(warming=True)
            if done:
                self.reset()

    def run_timestep(self, warming=False, render=False):
        start_state = self.env_state
        for state_processor in self.state_processors:
            start_state = state_processor(start_state, self.env)

        if warming:
            consumable_action = self.env.action_space.sample()
        else:
            pred = self.explorer.add_exploration(self.model.predict(start_state))
            consumable_action = self.pred_to_consumable(pred, self.env)
        learnable_action = self.consumable_to_learnable(consumable_action, self.env)

        reward = 0.
        for i in range(self.repeated_actions):
            if not warming and render:
                self.renders_by_episode[-1].append(self.env.render(mode='rgb_array'))
            self.env_state, r, done, info = self.env.step(consumable_action)
            reward += r
            if done: break

        if not warming:
            self.explorer.step(done)
            self.monitor.step(reward)

        if done:
            next_state = np.zeros(self.replay_buffer.state_buffer.shape[1:])
        else:
            next_state = self.env_state
            for state_processor in self.state_processors:
                next_state = state_processor(next_state, self.env)

        self.replay_buffer.add(start_state, learnable_action, reward, next_state, done)
        if not warming and self.replay_buffer.current_size >= self.batch_size:
            batch = self.replay_buffer.get_batch(self.batch_size)
            self.model.fit(**batch)

        return reward, done

    def run_episode(self, i, render=False):
        self.renders_by_episode.append([])
        total_reward = 0.
        while True:
            reward, done = self.run_timestep(render=render)
            total_reward += reward
            if done:
                self.monitor.write_summary()
                self.monitor.new_episode()
                break
        self.reset()
        if i % 10 == 0:
            r, d = self.monitor.get_stats()
            print(f"Episode: {i}. Average Reward: {r}. Average Duration: {d}. Explore: {self.explorer.explore_rate}")

    def run_indefinitely(self, report_freq=100, render=False):
        i = 0
        while True:
            self.run_episode(i, render)
            with open('/home/kerasgym/agents/keep_running.txt') as f:
                contents = f.read().strip()
                if contents == 'False':
                    print("Stopping.")
                    break
            i += 1
