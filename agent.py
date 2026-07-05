from rl_glue.rl_glue import BaseAgent


class Agent(BaseAgent):
    """
    Agent.
    """

    def __init__(self):
        """
        Declare agent variables.
        """
        pass

    def agent_init(self):
        """
        Start of a run.
        """
        pass

    def agent_start(self, state):
        """
        Start of an episode.

        Parameters
        ----------
        state : float64 numpy array with shape (state_dim,)
            the agent's current state

        Returns
        -------
        action : float64 numpy array with shape (action_dim,)
            the first action the agent takes
        """
        pass

    def agent_step(self, reward, next_state, terminal):
        """
        Take a step in the environment.

        Parameters
        ----------
        reward : float64
            the reward received for taking the last action
        next_state : float64 numpy array with shape (state_dim,)
            the agent's current state
        terminal : boolean
            true if the goal state has been reached after taking action; otherwise false

        Returns
        -------
        action : float64 numpy array with shape (action_dim,)
            the action the agent is taking
        """
        pass

    def agent_end(self, reward, next_state, terminal):
        """
        End of an episode.

        Parameters
        ----------
        reward : float64
            the reward received for entering the terminal state
        next_state : float64 numpy array with shape (state_dim,)
            the agent's current state
        terminal : boolean
            true if the goal state has been reached after taking action; otherwise false
        """
        pass

    def agent_message(self, message):
        """
        Receive a message from RLGlue.

        Parameters
        ----------
        message : str
            the message passed

        Returns
        -------
        response : str
            the agent's response to the message (optional)
        """
        pass

    def agent_close(self):
        """
        Close the agent.
        """
        pass
