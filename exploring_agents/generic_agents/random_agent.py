from typing import Final

from abstractions import Agent, RewardFunction, Action, DecayFunction, State, Effect, Q
from exploring_agents.policies.epsilon_greedy import EpsilonGreedy
from exploring_agents.policies.random_policy import RandomPolicy
from utils.policy import get_best_action_from_q, sample_action


class RandomAgent(Agent):
    def __init__(
        self, reward_function: RewardFunction, actions: list[Action], gamma: float = 1
    ):
        """
        Agent implementing a solution based on estimating the value of state-action pairs. It updates values after
        every action, by observing results and bootstrapping values from what is expected to be the best policy
        for the following state.

        :reward_function: the reward function we are trying to maximize
        :actions: actions available to the agent
        :gamma: the gamma discount value to be used when calculating episode returns
        """

        self.reward_function: Final = reward_function
        self.policy: RandomPolicy = RandomPolicy(actions)
        self.actions: Final = actions
        self.gamma = gamma

    def select_action(self, state: State) -> Action:
        """
        selects an action from a state based on the agent policy

        :param state: the state to select the action from
        :return: the selected action
        """

        return sample_action(self.policy, state, self.actions)

    def run_update(
        self, state: State, action: Action, effect: Effect, next_state: State
    ) -> float:
        reward = self.reward_function(effect)
        return reward

    def finalize_episode(
        self,
        episode_states: list[State],
        episode_returns: list[float],
        episode_actions: list[Action],
    ):
        pass
