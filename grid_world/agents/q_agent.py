from typing import Final, Collection

from grid_world.action import Action
from grid_world.grid_world import GridWorld
from grid_world.state import State
from grid_world.type_aliases import Policy, RewardFunction, Q
from grid_world.utils.evaluators import best_q_value
from grid_world.utils.policy import (
    get_random_policy,
    sample_action,
    get_e_greedy_policy,
)
from grid_world.utils.returns import returns_from_reward


class QAgent:
    def __init__(
        self,
        world: GridWorld,
        reward_function: RewardFunction,
        actions: Collection[Action] = None,
        policy: Policy = None,
        gamma: float = 1,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        q_0: Q = None,
    ):
        """
        Agent implementing a solution based on estimating the value of state-action pairs. It updates values after
        every action, by observing results and bootstrapping values from what is expected to be the best policy
        for the following state.

        :world: the world this agent will explore
        :reward_function: the reward function we are trying to maximize
        :actions: actions available to the agent
        :policy: initial policy for the agent
        :gamma: the gamma discount value to be used when calculating episode returns
        :alpha: learning rate
        :epsilon: exploration rate to be considered when building policies
        :q_0: initial estimates of state-action values, will be considered as a constant 0 if not provided
        """

        self.world: Final = world
        self.reward_function: Final = reward_function
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.policy = Policy if policy is not None else get_random_policy(self.actions)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q: Q = (
            q_0
            if q_0 is not None
            else {(s, a): 0 for s in self.world.states for a in self.actions}
        )
        self.visited_states: set[State] = set()

    def train(
        self,
        episodes: int = 100,
    ) -> tuple[list[int], list[float]]:
        episode_lengths = []
        episode_total_returns = []
        for _ in range(episodes):
            episode_actions, episode_states, episode_rewards = self.run_episode()
            episode_returns = returns_from_reward(episode_rewards, self.gamma)
            episode_lengths.append(len(episode_actions))
            episode_total_returns.append(episode_returns[0])

        return episode_lengths, episode_total_returns

    def run_episode(self, initial_state: State = None) -> [bool, int]:
        state = initial_state if initial_state is not None else self.world.initial_state

        episode_states = [state]
        episode_actions = []
        episode_rewards = []

        # run through the world while updating q and the policy as we go
        while state.kind != "terminal":
            action = sample_action(self.policy, state, self.actions)
            new_state, effect = self.world.take_action(state, action)
            reward = self.reward_function(effect)
            self.visited_states.add(new_state)

            # learn from what happened
            self.q[state, action] = self.q[state, action] + self.alpha * (
                reward
                + self.gamma * best_q_value(self.q, new_state, self.actions)
                - self.q[state, action]
            )

            # improve from what was learned
            self.policy = get_e_greedy_policy(
                self.q, self.visited_states, self.actions, self.epsilon
            )

            state = new_state
            episode_actions.append(action)
            episode_states.append(state)
            episode_rewards.append(reward)

        return episode_actions, episode_states, episode_rewards
