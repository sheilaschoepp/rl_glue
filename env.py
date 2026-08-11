"""
Abstract environment base class for RLGlue.
"""

from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    """
    Defines the interface of an RLGlue environment.

    ie. These methods must be defined in your own environment classes.
    """

    @abstractmethod
    def __init__(self):
        """
        Declare environment variables.
        """

    @abstractmethod
    def env_init(self, env_info=None):
        """
        Start of a run.
        """

    @abstractmethod
    def env_start(self):
        """
        Start of an episode.

        Returns
        -------
        state : Any
            the first state / observation of the environment
        """

    @abstractmethod
    def env_step(self, action):
        """
        A step taken by the environment.

        Parameters
        ----------
        action : Any
            the action taken by the agent

        Returns
        -------
        reward : float64
            the reward for taking action
        state : Any
            the state of the environment
        terminal : boolean
            true if the goal state has been reached after taking
            action; otherwise false
        """

    @abstractmethod
    def env_message(self, message):
        """
        Pass information to RLGlue.

        Parameters
        ----------
        message : str
            the message passed

        Returns
        -------
        response : str
            the environment's response to the message (optional)
        """

    @abstractmethod
    def env_cleanup(self):
        """
        Clean up the environment.
        """
