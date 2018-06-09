import numpy as np
import matplotlib.pyplot as plt
from copy import copy


class Schedule:
    def __init__(self, start_value, interval=-1):
        self.t = 0
        self.value = start_value
        self.interval = interval

    def _step(self):
        pass

    def step(self, new_episode):
        if self.interval > 0:
            if self.t % self.interval == 0:
                self._step()
        else:
            if new_episode:
                self._step()
        self.t += 1


class LinearDecay(Schedule):
    def __init__(self, start_value, decay_rate, interval=-1):
        self.decay_rate = decay_rate
        super().__init__(start_value, interval)

    def _step(self):
        self.value -= self.decay_rate

    def get(self):
        return self.value


class ExponentialDecay(Schedule):
    def __init__(self, start_value, decay_rate, interval=-1):
        self.decay_rate = decay_rate
        super().__init__(start_value, interval)

    def _step(self):
        self.value *= np.e ** (-self.decay_rate * self.t)

    def get(self):
        return self.value


class ScopingPeriodic(Schedule):
    def __init__(self, start_value=0.7, target_value=0.05, duration=1000, threshold=0.0001,
                 amp=0.2, period=0.1, interval=-1):
        self.start_value = start_value
        self.target_value = target_value
        self.decay_factor = ((1 - self.start_value) - np.log(threshold)) / duration
        self.amp = amp
        self.period = period
        self.restart_prob = 1. / duration
        super().__init__(start_value, interval)

    def _decay(self, t):
        return np.exp(-self.decay_factor * t + (self.start_value - 1)) + self.target_value

    def _periodic(self, t):
        return self.amp * np.sin(self.period * t) + 0.5

    def _step(self):
        self.value = self._decay(self.t) * (0.5 + self._periodic(self.t))
        if np.random.random() <= self.restart_prob:
            self.t = 0

    def get(self):
        return max([0., min([1., self.value])])


def graph_schedule(schedule, n_vals):
    this_schedule = copy(schedule)
    this_schedule.t = 0
    values = []
    for i in range(n_vals):
        values.append(this_schedule.get())
        this_schedule.step()
    plt.plot(np.arange(n_vals), values)
