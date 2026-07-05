"""
Glues together an experiment, agent, and environment.
"""

from abc import ABC, abstractmethod


class RLGlue:
    """
    Facilitates interaction between an agent and environment for reinforcement learning experiments.

    The RLGlue program mediates the communication between the agent and environment programs in response to commands
    from the experiment program.  -Brian Tanner & Adam White

    Parameters
    ----------
    env_obj : BaseEnvironment
        an object that implements BaseEnvironment
    agent_obj : BaseAgent
        an object that implements BaseAgent
    """

    def __init__(self, env_obj, agent_obj):
        self._environment = env_obj  # environment
        self._agent = agent_obj  # agent

        # useful statistics
        self._run_reward = None
        self._run_steps = None
        self._run_episodes = None
        self._episode_reward = None
        self._episode_steps = None

        self._action = None

    def run_reward(self):
        """
        Return the amount of reward accumulated in a single run.

        Returns
        -------
        run_reward : float64
            the amount of reward accumulated in a single run
        """
        return self._run_reward

    def episode_reward(self):
        """
        Return the amount of reward accumulated in an episode.

        Returns
        -------
        episode_reward : float64
            the amount of reward accumulated in an episode
        """
        return self._episode_reward

    def run_episodes(self):
        """
        Return the number of episodes in a single run.

        Returns
        -------
        run_episodes : int
            the number of episodes in a single run
        """
        return self._run_episodes

    def run_steps(self):
        """
        Return the number of steps in a single run.

        Returns
        -------
        run_steps : int
            the number of steps in a single run
        """
        return self._run_steps

    def episode_steps(self):
        """
        Return the number of steps in the current episode.

        Returns
        -------
        episode_steps : int
            the number of steps in the current episode
        """
        return self._episode_steps

    def rl_init(self, run_reward=0, run_steps=0, run_episodes=0):
        """
        Start of a run.

        Parameters
        ----------
        run_reward : float64
            the amount of reward accumulated in a single run
        run_steps : int
            the number of steps in a single run
        run_episodes : int
            the number of episodes in a single run
        """

        self._run_reward = (
            run_reward  # amount of reward accumulated in a single run
        )
        self._run_steps = run_steps  # number of steps in a single run
        self._run_episodes = run_episodes  # number of episodes in a single run
        self._episode_reward = 0  # amount of reward accumulated in an episode
        self._episode_steps = 0  # number of steps in the current episode

        self._action = None

        self._agent.agent_init()
        self._environment.env_init()

    # repeat for each episode
    def rl_start(self):
        """
        Start of an episode.

        Returns
        -------
        state : float64 numpy array with shape (state_dim,)
            the first state observation of the environment
        action : float64 numpy array with shape (action_dim,)
            the action selected by the agent
        """

        self._episode_reward = 0  # reward accumulated in an episode
        self._episode_steps = 0  # number of steps in the current episode
        # self._run_steps = max(self._run_steps, 0)  # number of steps in a run

        state = self._environment.env_start()
        self._action = self._agent.agent_start(state)

        return state, self._action

    def rl_step(self):
        """
        Take a step in the environment.

        Returns
        -------
        reward : float64
            the reward received for taking action
        next_state : float64 numpy array with shape (state_dim,)
            the state observation of the environment
        terminal : boolean
            true if the goal state has been reached after taking action; otherwise false
        action : float64 numpy array with shape (action_dim,)
            the action selected by the agent
        """
        reward, next_state, terminal = self._environment.env_step(
            self._action
        )  # returns reward, next_state, done

        self._episode_steps += 1
        self._run_steps += 1

        self._run_reward += reward
        self._episode_reward += reward

        if terminal:
            self._action = self._agent.agent_end(reward, next_state, terminal)
            self._run_episodes += 1
        else:
            self._action = self._agent.agent_step(reward, next_state, terminal)

        return reward, next_state, terminal, self._action

    def rl_episode(self, max_steps_this_episode=0):
        """
        Run an episode.

        Parameters
        ----------
        max_steps_this_episode : int
            the maximum number of steps that can be taken in the current episode (<=0 if no limit on number of steps)

        Returns
        -------
        terminal : boolean
            true if the goal state has been reached after taking action; otherwise false
        """

        terminal = False

        self.rl_start()

        while not terminal and (
            (max_steps_this_episode <= 0)
            or (self._episode_steps < max_steps_this_episode)
        ):
            _, _, terminal, _ = self.rl_step()

        return terminal

    # CONVENIENCE FUNCTIONS BELOW
    def rl_env_start(self):
        """
        Start of an episode.

        Useful when manually specifying agent actions (for debugging).

        Returns
        -------
        state : float64 numpy array with shape (state_dim,)
            the first state observation of the environment
        """
        self._episode_steps = 0

        return self._environment.env_start()

    def rl_env_step(self, action):
        """
        Take a step in the environment.

        Useful when manually specifying agent actions (for debugging).

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
        reward, state, terminal = self._environment.env_step(action)

        self._run_reward += reward

        if terminal:
            self._run_episodes += 1
        else:
            self._episode_steps += 1
            self._run_steps += 1

        return reward, state, terminal

    def rl_agent_message(self, message):
        """
        Pass a message to the agent.

        Parameters
        ----------
        message : str
            the message passed

        Returns
        -------
        response : str
            the agent's response
        """
        if message is None:
            message_to_send = ''
        else:
            message_to_send = message

        the_agent_response = self._agent.agent_message(message_to_send)
        if the_agent_response is None:
            the_agent_response = ''

        return the_agent_response

    def rl_env_message(self, message):
        """
        Pass a message to the environment.

        Parameters
        ----------
        message : str
            the message passed

        Returns
        -------
        response : str
            the environment's response
        """
        if message is None:
            message_to_send = ''
        else:
            message_to_send = message

        the_env_response = self._environment.env_message(message_to_send)
        if the_env_response is None:
            return ''

        return the_env_response


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
    def agent_init(self):
        """
        Start of a run.
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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


class BaseEnvironment(ABC):
    """
    Defines the interface of an RLGlue environment.

    ie. These methods must be defined in your own environment classes
    """

    @abstractmethod
    def __init__(self):
        """
        Declare environment variables.
        """

    @abstractmethod
    def env_init(self):
        """
        Start of a run.
        """

    @abstractmethod
    def env_start(self):
        """
        Start of an episode.

        Returns
        -------
        state : float64 numpy array with shape (state_dim,)
            the first state observation of the environment
        """

    @abstractmethod
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

    @abstractmethod
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
