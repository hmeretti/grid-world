from typing import Final

from abstractions import Agent, RewardFunction, Action, State, Effect
from policies import RandomPolicy
from utils.policy import sample_action


class RandomAgent(Agent):
    def __init__(
        self, reward_function: RewardFunction, actions: tuple[Action, ...], gamma: float = 1
    ):
        """
        Agent that chooses an action randomly for any state(from a uniform distribution).

        :param reward_function: the reward function we
        :param actions: actions available to the agent
        :param gamma: the gamma discount value to be used when calculating episode returns
        """

        self.reward_function: Final = reward_function
        self.policy: RandomPolicy = RandomPolicy(actions)
        self.actions: Final = actions
        self.gamma = gamma

    def select_action(self, state: State) -> Action:
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
