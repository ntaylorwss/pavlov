import numpy as np
import matplotlib.pyplot as plt
from copy import copy


class Schedule:
    """Base class for value schedule, used for decaying values in exploration methods.

    Parameters:
        start_value (float): initial value at first timestep.
        interval (int): update interval in number of timesteps (-1 means every episode).
                        default: -1.

    Member variables:
        t (int): current timestep index.
        value (float): current value.
        interval (int): update interval in number of timesteps.
    """
    def __init__(self, start_value, interval=-1):
        self.timestep = 0
        self.value = start_value
        self.interval = interval

    def _step(self):
        """Implement the logic of the schedule for one iteration of the value."""
        pass

    def step(self, new_episode):
        """Wrap the schedule logic to determine based on interval when to iterate value."""
        if self.interval > 0:
            if self.timestep % self.interval == 0:
                self._step()
        else:
            if new_episode:
                self._step()
        self.timestep += 1

    def get(self):
        """Get value, bounded in [0, 1]."""
        return max([0., min([1., self.value])])


class Constant(Schedule):
    """A schedule whose value never changes."""
    def __init__(self, start_value, interval=-1):
        super().__init__(start_value, interval)

    def _step(self):
        pass


class LinearDecay(Schedule):
    """A schedule whose value decreases linearly, before being capped at some value.

    Additional parameters:
        final_value (float): value to be capped at.
        num_steps (int): number of steps before value is capped.

    Additional member variables:
        per_step (float): amount to decrease by each step.
        num_steps (int): number of steps before value is capped.
    """
    def __init__(self, start_value, final_value, num_steps, interval=-1):
        super().__init__(start_value, interval)
        self.per_step = (start_value - final_value) / num_steps
        self.num_steps = num_steps

    def _step(self):
        if self.timestep < self.num_steps:
            self.value -= self.per_step


class ExponentialDecay(Schedule):
    """A schedule whose value decreases exponentially, to eventually asymptote at some value.
    
    Additional parameters:
        final_value (float): value to be capped at.
        decay_rate (float): rate of the exponential decay; the `a` in `e^(at)`.

    Additional member variables:
        final_value (float): value to be capped at.
        decay_rate (float): rate of the exponential decay; the `a` in `e^(at)`.
    """
    def __init__(self, start_value, final_value, decay_rate, interval=-1):
        # TODO: make it actually asymptote to final_value
        super().__init__(start_value, interval)
        self.final_value = final_value
        self.decay_rate = decay_rate

    def _step(self):
        self.value *= np.e ** (-self.decay_rate * self.timestep)


class ScopingPeriodic(Schedule):
    def __init__(self, start_value, target_value, duration, threshold=0.0001,
                 amp=0.2, period=0.1, interval=-1):
        super().__init__(start_value, interval)
        self.timesteparget_value = target_value
        self.decay_factor = ((1 - self.value) - np.log(threshold)) / duration
        self.amp = amp
        self.period = period
        self.restart_prob = 1. / duration

    def _decay(self, t):
        return np.exp(-self.decay_factor * t + (self.value - 1)) + self.timesteparget_value

    def _periodic(self, t):
        return self.amp * np.sin(self.period * t) + 0.5

    def _step(self):
        self.value = self._decay(self.timestep) * (0.5 + self._periodic(self.timestep))
        if np.random.random() <= self.restart_prob:
            self.timestep = 0


def graph_schedule(schedule, n_vals):
    """Create matplot graph of schedule values over a given number of timesteps.
    
    Parameters:
        schedule (pavlov.Schedule): the schedule to be examined.
        n_vals (int): number of values to visualize; the domain of the graph is [0, n_vals).
    """
    this_schedule = copy(schedule)
    this_schedule.interval = 1
    this_schedule.t = 0
    values = []
    for i in range(n_vals):
        values.append(this_schedule.get())
        this_schedule.step(False)
    plt.plot(np.arange(n_vals), values)
