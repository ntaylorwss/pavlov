from datetime import datetime
from .replay_buffer import ReplayBuffer


class Agent:
    """Defines interface of agents, implements run_episode and reset. run_timestep should be
       defined by the child class."""
    def __init__(self, env, state_reshaping_fns, model, action_processing_fn,
                 buffer_size, batch_size):
        self.i_episode = 0
        self.env = env
        self.env_state = self.env.reset()
        self.state_reshapers = state_reshaping_fns
        self.model = model
        self.action_processor = action_processing_fn
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # logging
        self.episode_outcomes = []
        self.episode_lengths = []

    def reset(self):
        self.env.reset()

    def run_timestep(self):
        state = self.env_state
        for state_reshaper in self.state_reshapers:
            state = state_reshaper(self.env, state)

        action = self.model.predict(state)
        consumable = self.action_processor(self.env, action)
        self.env_state, reward, done, info = self.env.step(consumable)

        next_state = self.env_state
        for state_reshaper in self.state_reshapers:
            next_state = state_reshaper(self.env, next_state)

        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.replay_buffer.current_size > self.batch_size:
            batch = self.replay_buffer.get_batch(self.batch_size)
            self.model.fit(**batch)

        return reward, done

    def run_episode(self):
        episode_length = 0
        while True:
            reward, done = self.run_timestep()
            episode_length += 1
            if done: break
            if episode_length > (self.env.n_rows * self.env.n_cols):
                raise ValueError("Game has gone on impossibly long.")
        self.episode_outcomes.append(reward)
        self.episode_lengths.append(episode_length)
        self.env_state = self.env.reset()
        self.i_episode += 1
