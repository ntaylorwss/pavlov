import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from custom_inherit import DocInheritMeta


class Schedule(metaclass=DocInheritMeta(style="numpy")):
    """Base class for value schedule, used for decaying values in exploration methods.

    Parameters
    ----------
    start_value : float
        initial value at first timestep.
    interval : int
        update interval in number of timesteps (default is -1, which means every episode).

    Attributes
    ----------
    t : int
        current timestep index.
    value : float
        current value.
    interval : int
        update interval in number of timesteps.
    """
    def __init__(self, start_value, interval=-1):
        self.timestep = 0
        self.value = start_value
        self.interval = interval

    def _step(self):
        """Implement the logic of the schedule for one iteration of the value."""
        pass

    def step(self, new_episode):
        """Wrap the schedule logic to determine based on interval when to iterate value.

        Parameters
        ----------
        new_episode : bool
            flag to indicate whether this step is one that resets the environment.
        """
        if self.interval > 0:
            if self.timestep % self.interval == 0:
                self._step()
        else:
            if new_episode:
                self._step()
        self.timestep += 1

    def get(self):
        """Get value, bounded in [0, 1].

        Returns
        -------
        float
            current value given by schedule.
        """
        return max([0., min([1., self.value])])


class ConstantSchedule(Schedule):
    """A schedule whose value never changes."""
    def __init__(self, start_value, interval=-1):
        super().__init__(start_value, interval)

    def _step(self):
        """Do nothing, since value never changes."""
        pass


class LinearDecaySchedule(Schedule):
    """A schedule whose value decreases linearly, before being capped at some value.

    Parameters
    ----------
    final_value : float
        value to be capped at.
    num_steps : int
        number of steps before value is capped.

    Attributes
    ----------
    per_step : float
        amount to decrease by each step.
    num_steps : int
        number of steps before value is capped.
    """
    def __init__(self, start_value, final_value, num_steps, interval=-1):
        super().__init__(start_value, interval)
        self.per_step = (start_value - final_value) / num_steps
        self.num_steps = num_steps

    def _step(self):
        """Subtract `per_step` from value."""
        if self.timestep < self.num_steps:
            self.value -= self.per_step


class ExponentialDecaySchedule(Schedule):
    """A schedule whose value decreases exponentially, to eventually asymptote at some value.

    Parameters
    ----------
    final_value : float
        value to be capped at.
    num_steps : int
        number of steps before value is capped.

    Attributes
    ----------
    num_steps : int
        number of steps before value is capped.
    final_value : float
        value to asymptote towards.
    """
    def __init__(self, start_value, final_value, num_steps, interval=-1):
        super().__init__(start_value, interval)
        self.start_value = start_value
        self.final_value = final_value
        self.num_steps = num_steps
        # calculate decay rate from available information
        self.decay_rate = np.exp(np.log(self.final_value / self.start_value) / self.num_steps)

    def _step(self):
        """Decay by the defined exponential factor."""
        new_value = self.decay_rate**(self.timestep) * self.start_value
        self.value = max([new_value, self.final_value])


class ScopingPeriodicSchedule(Schedule):
    """A schedule whose value decreases exponentially, but is also sinusoidal.

    Parameters
    ----------
    target_value : float
        value to asymptote towards.
    duration : int
        number of steps to get to `target_value`.
    threshold : float
        difference from `target_value` where it is considered reached.
    amp : float
        amplitude of the sinusoidal wave.
    period : float
        period of the sinusoidal wave.
    """

    def __init__(self, start_value, target_value, duration, threshold=0.0001,
                 amp=0.2, period=0.1, interval=-1):
        super().__init__(start_value, interval)
        self.target_value = target_value
        self.decay_factor = ((1 - self.value) - np.log(threshold)) / duration
        self.amp = amp
        self.period = period
        self.restart_prob = 1. / duration

    def _decay(self, t):
        return np.exp(-self.decay_factor * t + (self.value - 1)) + self.target_value

    def _periodic(self, t):
        return self.amp * np.sin(self.period * t) + 0.5

    def _step(self):
        self.value = self._decay(self.timestep) * (0.5 + self._periodic(self.timestep))
        if np.random.random() <= self.restart_prob:
            self.timestep = 0


def graph_schedule(schedule, n_vals):
    """Create matplot graph of schedule values over a given number of timesteps.

    Parameters
    ----------
    schedule : pavlov.Schedule
        the schedule to be examined.
    n_vals : int
        number of values to visualize; the domain of the graph is [0, n_vals).
    """
    this_schedule = copy(schedule)
    this_schedule.interval = 1
    this_schedule.t = 0
    values = []
    for i in range(n_vals):
        values.append(this_schedule.get())
        this_schedule.step(False)
    plt.plot(np.arange(n_vals), values)
