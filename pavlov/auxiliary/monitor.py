"""A monitor keeps track of metrics, runs tensorboard, and ensures model health.

Monitors become associated with Agents in a 1-to-1 relationship and collect information
about the Model that they're working with through the Agent.
They use this information to configure tensorboard as well as inspect the Model's health.

Model health includes, at this time, checking for NaN weights.
"""

import collections
import numpy as np
import tensorflow as tf


class Monitor:
    """Keeps track of metrics, run tensorboard, ensure model health.

    Prints average of previous episode metrics to stdout at defined interval.
    Writes logs for tensorboard to use for visualization of these metrics.

    Parameters
    ----------
    report_frequency : int
        interval for printing to stdout, in number of episodes.
    save_path : str
        file path for tensorboard logs.

    Attributes
    ----------
    agent : pavlov.Agent
        the agent the metrics are tracking.
    save_path : str
        file path for tensorboard logs
    report_frequency : int
        interval for printing to stdout, in number of episodes.
    rewards : collections.deque
        holds last `report_frequency` episodes' reward totals.
    durations : collections.deque
        holds last `report_frequency` episodes' durations.
    episode : int
        current episode number of agent.
    summary_placeholders : list of tensorflow ops
        tensorflow op for summaries.
    update_ops : list of tensorflow ops
        tensorflow op for summaries.
    summary_op : list of tensorflow ops
        tensorflow op for summaries.
    summary_writer : list of tensorflow ops
        tensorflow op for summaries.
    """
    def __init__(self, report_frequency, save_path):
        self.report_frequency = report_frequency
        self.save_path = save_path
        self.rewards = collections.deque([0], maxlen=report_frequency)
        self.durations = collections.deque([0], maxlen=report_frequency)
        self.episode = 0

    def _setup_summary(self):
        """Configure internal tensorflow ops for summary metrics."""
        episode_total_reward = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar('total_reward_episode', episode_total_reward)
        tf.summary.scalar('duration_episode', episode_duration)
        summary_vars = [episode_total_reward, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in summary_vars]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def configure(self, agent):
        """Associate monitor with agent, picking up information about its model.

        Parameters
        ----------
        agent : pavlov.Agent
            Agent to be monitored.
        """
        self.agent = agent
        self.summary_placeholders, self.update_ops, self.summary_op = self._setup_summary()
        self.summary_writer = tf.summary.FileWriter(self.save_path, agent.model.session.graph)

    def get_metrics(self):
        """Return averages over `report_frequency` window of the two metrics as tuple."""
        return (np.mean(self.rewards), np.mean(self.durations))

    def step(self, reward):
        """Add reward and increment duration for timestep.

        Parameters
        ----------
        reward : float
            reward for the episode, to be added to the log.
        """
        self.rewards[-1] += reward
        self.durations[-1] += 1

    def check_model_health(self):
        """Perform checks to ensure that the model hasn't hit an unrecoverable state.
        
        Raises
        ------
        ValueError
            raised if model is in 'unhealthy' state.
        """
        if self.agent.model.has_nan():
            msg = "Model has hit NaN weights: check #37 here: https://tinyurl.com/yahuwbno"
            raise ValueError(msg)

    def write_summary(self):
        """Write metrics to tensorflow logs."""
        stats = self.get_metrics()
        for i in range(len(stats)):
            self.agent.model.session.run(self.update_ops[i], feed_dict={
                self.summary_placeholders[i]: float(stats[i])
                })
            summary_str = self.agent.model.session.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, self.episode+1)

    def log_to_stdout(self):
        """Write metrics to stdout in formatted string."""
        avg_r = np.mean(self.rewards)
        avg_d = np.mean(self.durations)
        if self.episode > 0 and self.episode % self.report_frequency == 0:
            s = "End of episode {}. Last {} episodes: ".format(
                 self.episode, self.report_frequency)
            s += "Average reward: {}. ".format(avg_r)
            s += "Average duration: {}.".format(avg_d)
            print(s)

    def new_episode(self, do_logging):
        """Prepare for new episode.

        Tasks: check model health, log if required, write to tensorboard, start new metrics.

        Parameters
        ----------
        do_logging : bool
            indicate whether to log to stdout at the end of the episode.
        """
        self.check_model_health()
        if do_logging:
            self.log_to_stdout()
        self.write_summary()
        self.rewards.append(0)
        self.durations.append(0)
        self.episode += 1
