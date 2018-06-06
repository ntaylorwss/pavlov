# OpenAI Gym Agents with Keras
This will serve as a bit of a testing bed for different reinforcement learning algorithms, starting with the [Deep Deterministic Policy Gradient](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html). The main purpose is to build out a general project structure and interface for creating reinforcement learning agents, with modules such as:
- State processing pipeline functions
- Action processing functions (model output -> environment consumable)
- Exploration techniques, including schedulers for values such as the epsilon of Epsilon-Greedy
- A replay buffer
- A monitoring suite for various agent metrics and internals

The ultimate goal is that you will be able to mix and match these components with various algorithms and environments (based on the `step(action)` and `reset()` API defined by Gym environments) to quickly build and test reinforcement learning agents.
