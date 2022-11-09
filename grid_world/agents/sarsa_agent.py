from typing import Final, Collection

from grid_world.action import Action
from grid_world.grid_world import GridWorld
from grid_world.state import State
from grid_world.type_aliases import Policy, RewardFunction, Q
from grid_world.utils.policy import (
    get_random_policy,
    sample_action,
    get_e_greedy_policy,
)
from grid_world.utils.returns import returns_from_reward


class SarsaAgent:
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
        action = sample_action(self.policy, state, self.actions)
        while state.kind != "terminal":
            new_state, effect = self.world.take_action(state, action)
            next_action = sample_action(self.policy, new_state, self.actions)
            reward = self.reward_function(effect)
            self.visited_states.add(new_state)

            # learn from what happened
            self.q[state, action] = self.q[state, action] + self.alpha * (
                reward
                + self.gamma * self.q[new_state, next_action]
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
            action = next_action

        return episode_actions, episode_states, episode_rewards
