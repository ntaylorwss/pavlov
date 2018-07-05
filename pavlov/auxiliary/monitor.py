import collections
import numpy as np
import tensorflow as tf


class Monitor:
    """Keeps track of two metrics: reward and duration.

    Prints average of previous episode metrics to stdout at defined interval.
    Writes logs for tensorboard to use for visualization of these metrics.

    Parameters:
        agent (pavlov.Agent): the agent the metrics are tracking.
        save_path (str): file path for tensorboard logs.
                         default: /var/log/.
        report_freq (int): interval for printing to stdout, in number of episodes.

    Member variables:
        agent (pavlov.Agent): the agent the metrics are tracking.
        save_path (str): file path for tensorboard logs
        report_freq (int): interval for printing to stdout, in number of episodes.
        rewards (collections.deque): holds last `report_freq` episodes' reward totals.
        durations (collections.deque): holds last `report_freq` episodes' durations.
        episode (int): current episode number of agent.
        summary_placeholders ([tf object]): tensorflow op for summaries.
        update_ops ([tf object]): tensorflow op for summaries.
        summary_op (tf object): tensorflow op for summaries.
        summary_writer (tf object): tensorflow op for summaries.
    """
    def __init__(self, report_freq, save_path):
        self.save_path = save_path
        self.report_freq = report_freq
        self.rewards = collections.deque([0], maxlen=report_freq)
        self.durations = collections.deque([0], maxlen=report_freq)
        self.episode = 0

    def _setup_summary(self):
        """Configure internal tensorflow ops for summary metrics."""
        episode_total_reward = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(f'total_reward_episode', episode_total_reward)
        tf.summary.scalar(f'duration_episode', episode_duration)
        summary_vars = [episode_total_reward, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in summary_vars]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def configure(self, agent):
        self.agent = agent
        self.summary_placeholders, self.update_ops, self.summary_op = self._setup_summary()
        self.summary_writer = tf.summary.FileWriter(self.save_path, agent.model.session.graph)

    def get_metrics(self):
        """Return averages over `report_freq` window of the two metrics as tuple."""
        return (np.mean(self.rewards), np.mean(self.durations))

    def step(self, reward):
        """Add reward and increment duration for timestep."""
        self.rewards[-1] += reward
        self.durations[-1] += 1

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
        if self.episode % self.report_freq == 0:
            s = f"End of episode {self.episode+1}. Last {self.report_freq} episodes: "
            s += f"Average reward: {avg_r}. "
            s += f"Average duration: {avg_d}."
            print(s)

    def new_episode(self, do_logging):
        """Log if required, write to tensorboard, and Reset metric queues and move to next episode."""
        if do_logging:
            self.log_to_stdout()
        self.write_summary()
        self.rewards.append(0)
        self.durations.append(0)
        self.episode += 1
