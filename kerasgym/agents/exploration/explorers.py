import random
import numpy as np

invalid_fn = lambda action: raise TypeError("Incompatible space or prediction type.")


class Explorer:
    def __init__(self):
        self.explore_and_convert_fns = {
            'discrete': {'policy': self._discrete_policy,
                         'value': self._discrete_value},
            'multidiscrete': {'policy': self._multidiscrete_policy,
                              'value': self._multidiscrete_value},
            'box': {'policy': self._box_policy, 'value': self._box_value},
            'multibinary': {'policy': self._multibinary_policy,
                            'value': self._multibinary_value}}

    def configure(self, agent):
        s = str(type(agent.env.action_space))
        self.space_type = s[::-1][2:s[::-1].find('.')][::-1]
        self.pred_type = agent.model.pred_type
        self.agent = agent

    def explore_and_convert(self, pred):
        return self.add_exploration_fns[self.space_type][self.pred_type](pred)

    def step(self):
        pass


class EpsilonGreedy(Explorer):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
        self.epsilon = self.schedule.get()
        self.discrete = discrete

    def _discrete_policy(self, pred):
        if random.random() < self.epsilon:
            return self.agent.env.action_space.sample()
        else:
            return np.argmax(pred)

    def _discrete_value(self, pred):
        # policy and value are equivalent when argmaxing
        return _discrete_policy(pred)

    def _multidiscrete_policy(self, pred):
        if random.random() < self.epsilon:
            return self.agent.env.action_space.sample()
        else:
            return np.array(list(map(np.argmax, pred)))

    def _multidiscrete_value(self, pred):
        

    def _box_policy(self, pred):

    def _box_value(self, pred):

    def _multibinary_policy(self, pred):

    def _multibinary_value(self, pred):

    def add_exploration(self, pred):
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

    def add_exploration(self, pred):
        if random.random() < self.epsilon:
            pass
