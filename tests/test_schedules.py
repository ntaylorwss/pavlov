import os
os.chdir('/home')

from pavlov.auxiliary import schedules

eps = 0.00001

class Test_ConstantSchedule:
    def setup(self):
        self.schedule = schedules.ConstantSchedule(0.5, interval=1)

    def test_10_steps(self):
        for i in range(10):
            self.schedule.step(new_episode=False)
        assert self.schedule.get() == 0.5


class Test_LinearDecaySchedule:
    def setup(self):
        self.start_value = 0.9
        self.final_value = 0.1
        self.steps = 100
        self.schedule = schedules.LinearDecaySchedule(self.start_value, self.final_value,
                                                      self.steps, interval=1)

    def test_10_steps(self):
        for i in range(10):
            self.schedule.step(new_episode=False)
        target = self.start_value - ((self.start_value - self.final_value) / 100 * 10)
        diff = self.schedule.get() - target
        assert abs(diff) < eps

    def test_full_range(self):
        for i in range(self.steps):
            self.schedule.step(new_episode=False)
        diff = self.schedule.get() - self.final_value
        assert abs(diff) < eps


class Test_ExponentialDecaySchedule:
    def setup(self):
        self.start_value = 0.9
        self.final_value = 0.1
        self.steps = 50
        self.schedule = schedules.ExponentialDecaySchedule(self.start_value,
                                                           self.final_value, self.steps,
                                                           interval=1)

    def test_full_range(self):
        for i in range(self.steps):
            self.schedule.step(new_episode=False)
        diff = self.schedule.get() - self.final_value
        assert abs(diff) < 0.01
