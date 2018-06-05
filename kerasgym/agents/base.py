import numpy as np
import time
from datetime import datetime
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from .replay_buffer import ReplayBuffer


class Agent:
    """Defines interface of agents, implements run_episode and reset. run_timestep should be
       defined by the child class."""
    def __init__(self, env, state_processing_fns, model, action_processing_fn, explorer,
                 buffer_size, batch_size,
                 plt=None, ipy_display=None, is_learning=True):
        self.env = env
        self.env_state = self.env.reset()
        self.state_processors = state_processing_fns
        self.model = model
        self.action_processor = action_processing_fn
        self.explorer = explorer
        self.explorer.add_to_agent(self)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.is_learning = is_learning

        # display
        self.plt = plt
        self.ipy_display = ipy_display
        self.episode_frames = []

    def render_gif(self, episode=0):
        frames = self.episode_frames[episode]
        self.plt.figure(figsize=(frames[0].shape[1] / 72.0,
                                 frames[0].shape[0] / 72.0), dpi = 72)
        patch = self.plt.imshow(frames[0])
        self.plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(self.plt.gcf(), animate, frames = len(frames), interval=50)
        display(display_animation(anim, default_mode='loop'))

    def reset(self):
        self.env.reset()

    def run_timestep(self):
        self.episode_frames[-1].append(self.env.render(mode='rgb_array'))

        state = self.env_state
        for state_processor in self.state_processors:
            state = state_processor(state, self.env)

        action = self.explorer.add_exploration(self.model.predict(state))
        consumable = self.action_processor(action, self.env)

        self.env_state, reward, done, info = self.env.step(consumable)
        self.explorer.step()

        next_state = self.env_state
        for state_processor in self.state_processors:
            next_state = state_processor(next_state, self.env)

        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.is_learning and self.replay_buffer.current_size > self.batch_size:
            batch = self.replay_buffer.get_batch(self.batch_size)
            self.model.fit(**batch)

        return reward, done

    def run_episode(self):
        self.episode_frames.append([])
        while True:
            reward, done = self.run_timestep()
            if done: break
        self.env_state = self.env.reset()
