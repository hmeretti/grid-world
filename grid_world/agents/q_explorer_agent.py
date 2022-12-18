from typing import Final, Collection

from grid_world.action import Action
from grid_world.utils.world_map import WorldMap
from grid_world.grid_world import GridWorld
from grid_world.type_aliases import Policy, RewardFunction, Q
from grid_world.utils.evaluators import best_q_value
from grid_world.utils.policy import (
    get_random_policy,
    sample_action,
    get_explorer_policy,
)
from grid_world.utils.returns import returns_from_reward


class QExplorerAgent:
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: Collection[Action] = None,
        policy: Policy = None,
        gamma: float = 1,
        alpha: float = 0.1,
        epsilon: float = 0.1,
    ):
        """
        Agent very similar to q-learning, that uses an improved exploration policy. The agent uses the same method
        as q-learning to update its Q function, however it uses a different exploration policy, more suited to this
        specific problem. It does so by keeping track of a world map, and ignoring actions that are surely bad when
        doing exploration(hitting walls or traps).

        This agent is not well suited for stochastic worlds.

        :reward_function: the reward function we are trying to maximize
        :actions: actions available to the agent
        :policy: initial policy for the agent
        :gamma: the gamma discount value to be used when calculating episode returns
        :alpha: learning rate
        :epsilon: exploration rate to be considered when building policies

        """
        self.reward_function: Final = reward_function
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.policy = Policy if policy is not None else get_random_policy(self.actions)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q: Q = dict()
        self.world_map: WorldMap = WorldMap(world_states=set(), actions=self.actions)

    def train(
        self,
        world: GridWorld,
        episodes: int = 100,
    ) -> tuple[list[int], list[float]]:
        episode_lengths = []
        episode_total_returns = []
        for _ in range(episodes):
            episode_actions, episode_states, episode_rewards = self.run_episode(world)
            episode_returns = returns_from_reward(episode_rewards, self.gamma)
            episode_lengths.append(len(episode_actions))
            episode_total_returns.append(episode_returns[0])

        return episode_lengths, episode_total_returns

    def run_episode(self, world: GridWorld) -> [bool, int]:
        state = world.initial_state

        episode_states = [state]
        episode_actions = []
        episode_rewards = []

        # run through the world while updating q the policy and our map as we go
        effect = 0
        while effect != 1:
            action = sample_action(self.policy, state, self.actions)
            new_state, effect = world.take_action(state, action)
            reward = self.reward_function(effect)

            # update our map based on what happened
            self.world_map.update_map(state, action, new_state)

            # learn from what happened
            cur_q = self.q.get((state, action), 0)
            self.q[state, action] = cur_q + self.alpha * (
                reward
                + self.gamma
                * best_q_value(
                    self.q,
                    new_state,
                    self.world_map.reasonable_actions.get(new_state, self.actions),
                )
                - cur_q
            )

            # improve from what was learned
            self.policy = get_explorer_policy(
                self.q,
                self.world_map.world_states,
                self.actions,
                self.world_map.reasonable_actions,
                self.epsilon,
            )

            state = new_state
            episode_actions.append(action)
            episode_states.append(state)
            episode_rewards.append(reward)
        return episode_actions, episode_states, episode_rewards
