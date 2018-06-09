import tensorflow as tf


class MetricMonitor:
    def __init__(self, agent, save_path):
        self.agent = agent
        self.total_reward = 0.
        self.duration = 0
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

    def _get_stats(self):
        return [self.total_reward, self.duration]

    def step(self, reward):
        self.total_reward += reward
        self.duration += 1

    def write_summary(self):
        stats = self._get_stats()
        for i in range(len(stats)):
            self.agent.model.session.run(self.update_ops[i], feed_dict={
                self.summary_placeholders[i]: float(stats[i])
                })
            summary_str = self.agent.model.session.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, self.episode+1)

    def new_episode(self):
        self.total_reward = 0.
        self.duration = 0
        self.episode += 1
