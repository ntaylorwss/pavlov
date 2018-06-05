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
    def __init__(self, schedule):
        super().__init__()
        self.returns_processed = True
        self.schedule = schedule
        self.epsilon = self.schedule.get()
        self.n_explores = 0
        self.n_actions = 0

    def add_exploration(self, action):
        self.n_actions += 1
        if random.random() < self.epsilon:
            self.n_explores += 1
            action_num = self.agent.env.action_space.sample()
            action = np.zeros(self.agent.env.action_space.n)
            action[action_num] = 1
        return action

    def step(self):
        self.schedule.step()
        self.epsilon = self.schedule.get()

    @property
    def explore_rate(self):
        return self.n_explores / self.n_actions
