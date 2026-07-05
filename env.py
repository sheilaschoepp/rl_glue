from rl_glue.rl_glue import BaseEnvironment


class Environment(BaseEnvironment):
    """
    Environment.
    """

    def __init__(self):
        """
        Declare environment variables.

        Parameters
        ----------
        seed : int
            the random seed for the environment
        """
        pass

    # repeat for each run
    def env_init(self):
        """
        Start of a run.
        """
        pass

    # repeat for each episode
    def env_start(self):
        """
        Start of an episode.

        Returns
        -------
        state : float64 numpy array with shape (state_dim,)
            the first state observation of the environment
        """
        pass

    def env_step(self, action):
        """
        Take a step in the environment.

        Parameters
        ----------
        action : float64 numpy array with shape (action_dim,)
            the action taken by the agent

        Returns
        -------
        reward : float64
            the reward received for taking action
        state : float64 numpy array with shape (state_dim,)
            the state observation of the environment
        terminal : boolean
            true if the goal state has been reached after taking action; otherwise false
        """
        pass

    def env_message(self, message):
        """
        Receive a message from RLGlue.

        Parameters
        ----------
        message : str
            the message passed

        Returns
        -------
        response : str
            the environment's response to the message (optional)
        """
        pass

    def env_close(self):
        """
        Close the environment.
        """
        pass
