"""
Glues together an experiment, agent, and environment.
"""


class RLGlue:
    """
    Facilitates interaction between an agent and environment for
    reinforcement learning experiments.

    The RLGlue program mediates the communication between the agent and
    environment programs in response to commands from the experiment
    program.  -Brian Tanner & Adam White

    Parameters
    ----------
    env_obj : BaseEnvironment
        an object that implements BaseEnvironment
    agent_obj : BaseAgent
        an object that implements BaseAgent
    """

    def __init__(self, env_obj, agent_obj):

        self._environment = env_obj
        self._agent = agent_obj

        self._run_episodes = None
        self._run_steps = None
        self._run_reward = None

        self._episode_steps = None
        self._episode_reward = None

        self._action = None

    def rl_init(
        self,
        agent_info=None,
        env_info=None,
        run_reward=0.0,
        run_steps=0,
        run_episodes=0,
    ):
        """
        Start of an RLGlue run.

        Parameters
        ----------
        env_info : dict or None
            information for environment initialization
        agent_info : dict or None
            information for agent initialization
        run_reward : float64
            the amount of reward accumulated in a single run
        run_steps : int
            the number of steps in a single run
        run_episodes : int
            the number of episodes in a single run
        """
        if env_info is None:
            env_info = {}

        if agent_info is None:
            agent_info = {}

        self._run_episodes = run_episodes
        self._run_steps = run_steps
        self._run_reward = run_reward

        self._episode_steps = None
        self._episode_reward = None

        self._action = None

        self._environment.env_init(env_info)
        self._agent.agent_init(agent_info)

    def rl_start(self):
        """
        Start of an RLGlue episode.

        Returns
        -------
        state : Any
            the first state of the environment
        action : Any
            the action selected by the agent
        """
        self._episode_steps = 0
        self._episode_reward = 0.0

        state = self._environment.env_start()
        self._action = self._agent.agent_start(state)

        return state, self._action

    def rl_step(self):
        """
        A step taken by RLGlue.

        Returns
        -------
        reward : float64
            the reward received for taking action
        next_state : Any
            the state observation of the environment
        action : Any or None
            the action selected by the agent or None if the episode
            terminated
        terminal : boolean
            true if the goal state has been reached after taking
            action; otherwise false
        """
        reward, next_state, terminal = self._environment.env_step(self._action)

        self._run_steps += 1
        self._episode_steps += 1

        self._run_reward += reward
        self._episode_reward += reward

        if terminal:
            self._agent.agent_end(reward)
            self._action = None
            self._run_episodes += 1
        else:
            self._action = self._agent.agent_step(reward, next_state)

        return reward, next_state, self._action, terminal

    def rl_episode(self, max_steps_this_episode=0):
        """
        Run an RLGlue episode.

        Parameters
        ----------
        max_steps_this_episode : int
            the maximum number of steps that can be taken in the
            current episode (<=0 if no limit on number of steps)

        Returns
        -------
        terminal : boolean
            true if the goal state has been reached after taking
            action; otherwise false
        """
        terminal = False

        self.rl_start()

        while not terminal and (
            max_steps_this_episode <= 0
            or self._episode_steps < max_steps_this_episode
        ):
            _, _, _, terminal = self.rl_step()

        return terminal

    def rl_agent_message(self, message):
        """
        Pass information to the agent.

        Parameters
        ----------
        message : str
            the message passed

        Returns
        -------
        response : str
            the agent's response
        """
        response = self._agent.agent_message(message)

        return response

    def rl_env_message(self, message):
        """
        Pass information to the environment.

        Parameters
        ----------
        message : str
            the message passed

        Returns
        -------
        response : str
            the environment's response
        """
        response = self._environment.env_message(message)

        return response

    def rl_cleanup(self):
        """
        Clean up the environment and agent.
        """
        self._environment.env_cleanup()
        self._agent.agent_cleanup()
        self._action = None

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
