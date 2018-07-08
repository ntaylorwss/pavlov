"""Actors add exploration to model predictions and convert them to the appropriate format for acting.

Actors become associated with Agents in a 1-to-1 relationship and collect information
about the Model and Environment that they're working with through the Agent.
They use this information to convert predictions to the appropriate format,
such as converting a policy vector from a Policy model to an integer for a Discrete action space.

Often, an Actor will require at least one Schedule to govern a particular value, such as
epsilon for the epsilon-greedy approach.

Currently implemented exploration methods are:
- Epsilon greedy.
"""

import numpy as np
from custom_inherit import DocInheritMeta
from ..util import ActionModelMismatchError


class Actor(metaclass=DocInheritMeta(style="numpy")):
    """Component responsible both for exploration and converting model predictions to actions.

    Returns 2 actions at a time: the first to be consumed by the model,
                                 the second to be consumed by the environment.

    Attributes
    -------
    explore_and_convert_fns : dict of {str : dict of {str : function}}
        Functions corresponding to every combination of action space and model type.
        Action space is the first key, model type is the second key.
        The functions add exploration and convert the chosen action to the correct format.
    action_space : gym.Space
        The action space object of the associated environment.
    action_type : str
        The class name of the action space as a lowercased string.
    prediction_type : {'value', 'policy'}
        Indicates whether the model outputs action-values or a policy vector for the state.
    """
    def __init__(self):
        # a dictionary of functions makes choosing the method of conversion easier
        self.explore_and_convert_fns = {
            'discrete': {'policy': self._discrete_policy,
                         'value': self._discrete_value},
            'multidiscrete': {'policy': self._multidiscrete_policy,
                              'value': self._multidiscrete_value},
            'box': {'policy': self._box_policy, 'value': self._box_value},
            'multibinary': {'policy': self._multibinary_policy,
                            'value': self._multibinary_value}}
        self.action_space = None
        self.action_type = None
        self.prediction_type = None

    def configure(self, agent):
        """Associate actor with agent, picking up information about its action space and model."""
        self.action_space = agent.env.action_space
        self.action_type = self.action_space.__class__.__name__.lower()
        self.prediction_type = agent.model.prediction_type

    def convert_pred(self, pred):
        """Look up explore_and_convert function and apply it to model prediction."""
        return self.explore_and_convert_fns[self.action_type][self.prediction_type](pred)

    def warming_action(self):
        """Choose a random action without involving the model.

        Returns
        -------
        action_for_model : numpy.ndarray
            The action that will be consumed by the model for learning.
        action_for_env : numpy.ndarray or int or float
            The action that will be consumed by the environment to step.
        """
        action_for_env = self.action_space.sample()
        if self.action_type == 'discrete':
            action_for_model = np.eye(self.action_space.n)[action_for_env]
        elif self.action_type == 'multidiscrete':
            action_for_model = [np.eye(n)[action_for_env[i]]
                           for i, n in enumerate(self.action_space.nvec)]
        elif self.action_type == 'box':
            action_for_model = action_for_env
        elif self.action_type == 'multibinary':
            action_for_model = [np.eye(2)[env_a] for env_a in action_for_env]
        return action_for_model, action_for_env

    def step(self, new_episode):
        """Move one timestep ahead. Main purpose is to advance value schedules.

        Parameters
        ----------
        new_episode : bool
            A flag indicating whether this step is one that resets the environment.
        """
        pass

    def _discrete_policy(self, pred):
        """Exploration and conversion function for a Discrete action space + Policy model.

        Parameters
        ----------
        pred : numpy.ndarray
            The prediction output by the model, to be converted.

        Raises
        ------
        NotImplementedError.
        """
        raise NotImplementedError

    def _discrete_value(self, pred):
        """Exploration and conversion function for a Discrete action space + Value model.

        Parameters
        ----------
        pred : numpy.ndarray
            The prediction output by the model, to be converted.

        Raises
        ------
        NotImplementedError.
        """
        raise NotImplementedError

    def _multidiscrete_policy(self, pred):
        """Exploration and conversion function for a Multi-Discrete action space + Policy model.

        Parameters
        ----------
        pred : numpy.ndarray
            The prediction output by the model, to be converted.

        Raises
        ------
        NotImplementedError.
        """
        raise NotImplementedError

    def _multidiscrete_value(self, pred):
        """Exploration and conversion function for a Multi-Discrete action space + Value model.

        Parameters
        ----------
        pred : numpy.ndarray
            The prediction output by the model, to be converted.

        Raises
        ------
        NotImplementedError.
        """
        raise NotImplementedError

    def _box_policy(self, pred):
        """Exploration and conversion function for a Box action space + Policy model.

        Parameters
        ----------
        pred : numpy.ndarray
            The prediction output by the model, to be converted.

        Raises
        ------
        NotImplementedError.
        """
        raise NotImplementedError

    def _box_value(self, pred):
        """Exploration and conversion function for a Box action space + Value model.

        Parameters
        ----------
        pred : numpy.ndarray
            The prediction output by the model, to be converted.

        Raises
        ------
        NotImplementedError.
        """
        raise NotImplementedError

    def _multibinary_policy(self, pred):
        """Exploration and conversion function for a Multi-Binary action space + Policy model.

        Parameters
        ----------
        pred : numpy.ndarray
            The prediction output by the model, to be converted.

        Raises
        ------
        NotImplementedError.
        """
        raise NotImplementedError

    def _multibinary_value(self, pred):
        """Exploration and conversion function for a Multi-Binary action space + Value model.

        Parameters
        ----------
        pred : numpy.ndarray
            The prediction output by the model, to be converted.

        Raises
        ------
        NotImplementedError.
        """
        pass


class EpsilonGreedyActor(Actor):
    """Takes a random action with a varying probability, otherwise does what it considers optimal.

    Parameters
    ----------
    epsilon_schedule : pavlov.auxiliary.Schedule
        The schedule that governs the value of epsilon.

    Attributes
    ----------
    epsilon_schedule : pavlov.auxiliary.Schedule
        The schedule that governs the value of epsilon.
    epsilon : float
        The current value of epsilon to be applied.
    """
    def __init__(self, epsilon_schedule):
        super().__init__()
        self.epsilon_schedule = epsilon_schedule
        self.epsilon = self.epsilon_schedule.get()

    def step(self, new_episode):
        """Iterate the epsilon value according to its schedule."""
        self.epsilon_schedule.step(new_episode)
        self.epsilon = self.epsilon_schedule.get()

    def _discrete_policy(self, pred):
        """Apply epsilon-greedy to discrete action space with a policy-based model.

        Returns
        -------
        action_for_model : numpy.ndarray
            The action that will be consumed by the model for learning.
        action_for_env : numpy.ndarray or int or float
            The action that will be consumed by the environment to step.
        """
        if np.random.random() < self.epsilon:
            action_for_env = self.action_space.sample()
        else:
            action_for_env = np.argmax(pred)
        action_for_model = np.eye(self.action_space.n)[action_for_env]
        return action_for_model, action_for_env

    def _discrete_value(self, pred):
        """Apply epsilon-greedy to discrete action space with a value-based model.

        Returns
        -------
        action_for_model : numpy.ndarray
            The action that will be consumed by the model for learning.
        action_for_env : numpy.ndarray or int or float
            The action that will be consumed by the environment to step.
        """
        return self._discrete_policy(pred)

    def _multidiscrete_policy(self, pred):
        """Apply epsilon-greedy to multidiscrete action space with a policy-based model.

        Returns
        -------
        action_for_model : numpy.ndarray
            The action that will be consumed by the model for learning.
        action_for_env : numpy.ndarray or int or float
            The action that will be consumed by the environment to step.
        """
        if np.random.random() < self.epsilon:
            action_for_env = self.action_space.sample()
        else:
            action_for_env = np.array(list(map(np.argmax, pred)))
        action_for_model = [np.eye(n)[action_for_env[i]]
                       for i, n in enumerate(self.action_space.nvec)]
        return action_for_model, action_for_env

    def _multidiscrete_value(self, pred):
        """Apply epsilon-greedy to multidiscrete action space with a value-based model.

        Returns
        -------
        action_for_model : numpy.ndarray
            The action that will be consumed by the model for learning.
        action_for_env : numpy.ndarray or int or float
            The action that will be consumed by the environment to step.
        """
        return self._multidiscrete_policy(pred)

    def _box_policy(self, pred):
        """Apply epsilon-greedy to box action space with a policy-based model.

        Returns
        -------
        action_for_model : numpy.ndarray
            The action that will be consumed by the model for learning.
        action_for_env : numpy.ndarray or int or float
            The action that will be consumed by the environment to step.
        """
        if np.random.random() < self.epsilon:
            both_actions = self.action_space.sample()
            return both_actions, both_actions
        else:
            return pred, pred

    def _box_value(self, pred):
        """Apply epsilon-greedy to box action space with a value-based model.

        Returns
        -------
        action_for_model : numpy.ndarray
            The action that will be consumed by the model for learning.
        action_for_env : numpy.ndarray or int or float
            The action that will be consumed by the environment to step.
        """
        raise ActionModelMismatchError('box', 'value')

    def _multibinary_policy(self, pred):
        """Apply epsilon-greedy to multibinary action space with a policy-based model.

        Returns
        -------
        action_for_model : numpy.ndarray
            The action that will be consumed by the model for learning.
        action_for_env : numpy.ndarray or int or float
            The action that will be consumed by the environment to step.
        """
        return self._discrete_policy(pred)

    def _multibinary_value(self, pred):
        """Apply epsilon-greedy to multibinary action space with a value-based model.

        Returns
        -------
        action_for_model : numpy.ndarray
            The action that will be consumed by the model for learning.
        action_for_env : numpy.ndarray or int or float
            The action that will be consumed by the environment to step.
        """
        return self._discrete_policy(pred)


'''
class EpsilonNoisyActor(Actor):
    def __init__(self, eps_schedule, mu_schedule, sigma_schedule):
        super.__init__()
        self.eps_schedule = eps_schedule
        self.mu_schedule = mu_schedule
        self.sigma_schedule = sigma_schedule
        self.epsilon = self.eps_schedule.get()
        self.mu = self.mu_schedule.get()
        self.sigma = self.sigma_schedule.get()

    # TODO: this whole thing
'''
