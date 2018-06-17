import random
import numpy as np


class Explorer:
    def __init__(self):
        self.agent = None

    def add_to_agent(self, agent):
        self.agent = agent

    def add_exploration(self, action):
        return action

    def step(self):
        pass


class EpsilonGreedy(Explorer):
    def __init__(self, schedule, discrete=True):
        super().__init__()
        self.schedule = schedule
        self.epsilon = self.schedule.get()
        self.discrete = discrete

    def add_exploration(self, action):
        if random.random() < self.epsilon:
            action = self.agent.env.action_space.sample()
            if self.discrete:
                ohe_action = np.zeros(self.agent.model.action_dim)
                ohe_action[action] = 1
                return ohe_action
        return action

    def step(self, new_episode):
        self.schedule.step(new_episode)
        self.epsilon = self.schedule.get()

    @property
    def explore_rate(self):
        return round(self.schedule.get(), 3)


class EpsilonNoisy(Explorer):
    def __init__(self, eps_schedule, mu_schedule, sigma_schedule):
        super.__init__()
        self.eps_schedule = schedule
        self.mu_schedule = mu_schedule
        self.sigma_schedule = sigma_schedule
        self.epsilon = self.eps_schedule.get()
        self.mu = self.mu_schedule.get()
        self.sigma = self.sigma_schedule.get()

    def add_exploration(self, action):
        if random.random() < self.epsilon:
            pass
