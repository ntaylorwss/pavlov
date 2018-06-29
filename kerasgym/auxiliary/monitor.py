import collections.deque
import numpy as np
import tensorflow as tf


class Monitor:
    def __init__(self, agent, save_path, report_freq=10):
        self.agent = agent
        self.rewards = collections.deque([0], maxlen=report_freq)
        self.durations = collections.deque([0], maxlen=report_freq)
        self.report_freq = report_freq
        self.episode = 0
        self.save_path = save_path
        self.summary_placeholders, self.update_ops, self.summary_op = self._setup_summary()
        self.summary_writer = tf.summary.FileWriter(self.save_path, self.agent.model.session.graph)

    def _setup_summary(self):
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

    def get_stats(self):
        return [np.mean(self.rewards), np.mean(self.durations)]

    def step(self, reward):
        self.rewards[-1] += reward
        self.durations[-1] += 1

    def write_summary(self):
        stats = self.get_stats()
        for i in range(len(stats)):
            self.agent.model.session.run(self.update_ops[i], feed_dict={
                self.summary_placeholders[i]: float(stats[i])
                })
            summary_str = self.agent.model.session.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, self.episode+1)

    def log_to_stdout(self):
        avg_r = np.mean(self.rewards)
        avg_d = np.mean(self.durations)
        if self.episode > 0 and self.episode % self.report_freq == 0:
            s = f"End of episode {self.episode}. Last {self.report_freq} episodes: "
            s += f"Average reward: {avg_r}. "
            s += f"Average duration: {avg_d}."
            print(s)

    def new_episode(self):
        self.rewards.append(0)
        self.durations.append(0)
        self.episode += 1
