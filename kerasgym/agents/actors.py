import random
import numpy as np
from ..util import get_action_type, ActionModelMismatchError


class Actor:
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
        self.action_space = agent.env.action_space
        self.action_type = get_action_type(self.action_space)
        self.pred_type = agent.model.pred_type

    def convert_pred(self, pred):
        return self.explore_and_convert_fns[self.action_type][self.pred_type](pred)

    def warming_action(self):
        a_for_env = self.action_space.sample()
        if self.pred_type == 'discrete':
            a_for_model = np.eye(self.action_space.n)[a_for_env]
        elif self.pred_type == 'multidiscrete':
            a_for_model = [np.eye(n)[a_for_env[i]]
                           for i, n in enumerate(self.action_space.nvec)]
        elif self.pred_type == 'box':
            a_for_model = a_for_env
        elif self.pred_type == 'multibinary':
            a_for_model = [np.eye(2)[env_a] for env_a in a_for_env]
        return a_for_model, a_for_env

    def step(self):
        pass


class EpsilonGreedyActor(Actor):
    def __init__(self, epsilon_schedule):
        super().__init__()
        self.epsilon_schedule = epsilon_schedule
        self.epsilon = self.epsilon_schedule.get()

    def _discrete_policy(self, pred):
        if random.random() < self.epsilon:
            a_for_env = self.action_space.sample()
        else:
            a_for_env = np.argmax(pred)
        a_for_model = np.eye(self.action_space.n)[a_for_env]
        return a_for_model, a_for_env

    def _discrete_value(self, pred):
        return self._discrete_policy(pred)

    def _multidiscrete_policy(self, pred):
        if random.random() < self.epsilon:
            a_for_env = self.action_space.sample()
        else:
            a_for_env = np.array(list(map(np.argmax, pred)))
        a_for_model = [np.eye(n)[a_for_env[i]]
                       for i, n in enumerate(self.action_space.nvec)]
        return a_for_model, a_for_env

    def _multidiscrete_value(self, pred):
        return self._multidiscrete_policy(pred)

    def _box_policy(self, pred):
        if random.random() < self.epsilon:
            both_actions = self.action_space.sample()
            return both_actions, both_actions
        else:
            return pred, pred

    def _box_value(self, pred):
        raise ActionModelMismatchError('box', 'value')

    def _multibinary_policy(self, pred):
        return self._discrete_policy(pred)

    def _multibinary_value(self, pred):
        return self._discrete_policy(pred)

    def step(self, new_episode):
        self.epsilon_schedule.step(new_episode)
        self.epsilon = self.epsilon_schedule.get()

    @property
    def explore_rate(self):
        return round(self.epsilon_schedule.get(), 3)


class EpsilonNoisyActor(Actor):
    def __init__(self, eps_schedule, mu_schedule, sigma_schedule):
        super.__init__()
        self.eps_schedule = eps_schedule
        self.mu_schedule = mu_schedule
        self.sigma_schedule = sigma_schedule
        self.epsilon = self.eps_schedule.get()
        self.mu = self.mu_schedule.get()
        self.sigma = self.sigma_schedule.get()
