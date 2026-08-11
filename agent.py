"""
An abstract class that specifies the Agent for RLGlue.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Defines the interface of an RLGlue Agent.

    ie. These methods must be defined in your own Agent classes
    """

    @abstractmethod
    def __init__(self):
        """
        Declare agent variables.
        """

    @abstractmethod
    def agent_init(self, agent_info=None):
        """
        Start of a run.
        """

    @abstractmethod
    def agent_start(self, state):
        """
        Start of an episode.

        Parameters
        ----------
        state : Any
            the agent's state

        Returns
        -------
        action : Any
            the first action the agent takes
        """

    @abstractmethod
    def agent_step(self, reward, next_state):
        """
        A step taken by the agent.

        Parameters
        ----------
        reward : float64
            the reward the agent received for taking the last action
        next_state : Any
            the agent's state after taking the last action

        Returns
        -------
        action : Any
            the action the agent is taking
        """

    @abstractmethod
    def agent_end(self, reward):
        """
        End of an episode.

        Parameters
        ----------
        reward : float64
            the reward the agent received for entering the terminal
            state
        """

    @abstractmethod
    def agent_message(self, message):
        """
        Pass information to RLGlue.

        Parameters
        ----------
        message : str
            the message passed

        Returns
        -------
        response : str
            the agent's response to the message (optional)
        """

    @abstractmethod
    def agent_cleanup(self):
        """
        Clean up the agent.
        """
