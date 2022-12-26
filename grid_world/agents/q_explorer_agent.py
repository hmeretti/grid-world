from typing import Final, Collection

from grid_world.action import Action
from grid_world.agents.policies.epsilon_explorer import EpsilonExplorer
from grid_world.state import State
from grid_world.agents.commons.world_map import WorldMap
from grid_world.grid_world import GridWorld
from grid_world.type_aliases import Policy, RewardFunction, Q, DecayFunction
from grid_world.utils.evaluators import best_q_value
from grid_world.utils.policy import (
    sample_action,
    get_best_action_from_dict,
)
from grid_world.utils.returns import returns_from_reward


class QExplorerAgent:
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: Collection[Action] = None,
        gamma: float = 1,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: DecayFunction = None,
        alpha_decay: DecayFunction = None,
        q_0: Q = None,
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
        :q_0: initial estimates of state-action values, will be considered as a constant 0 if not provided

        """
        self.reward_function: Final = reward_function
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.policy: EpsilonExplorer = EpsilonExplorer(epsilon, actions, epsilon_decay)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q: Q = q_0 if q_0 is not None else {}
        self.alpha_decay = alpha_decay if alpha_decay is not None else (lambda x: x)
        self.world_map: WorldMap = WorldMap(
            world_states=set(x for (x, a) in self.q.keys()), actions=self.actions
        )
        self.policy_map: dict[[State, Action], float] = {}

    def train(
        self,
        world: GridWorld,
        episodes: int = 100,
    ) -> tuple[list[int], list[float]]:
        episode_lengths = []
        episode_total_returns = []
        for _ in range(episodes):
            episode_states, episode_rewards, _ = self.run_episode(world)
            episode_returns = returns_from_reward(episode_rewards, self.gamma)
            episode_lengths.append(len(episode_states))
            episode_total_returns.append(episode_returns[0])
            self.policy.decay()
            self.alpha = self.alpha_decay(self.alpha)

        return episode_lengths, episode_total_returns

    def run_episode(
        self, world: GridWorld, initial_state: State = None
    ) -> tuple[list[State], list[float], list[Action]]:
        state = initial_state if initial_state is not None else world.initial_state
        self.world_map.world_states.add(state)

        episode_states = []
        episode_rewards = []
        episode_actions = []

        # run through the world while updating q the policy and our map as we go
        effect = 0
        while effect != 1:
            action = sample_action(
                self.policy,
                state,
                self.world_map.reasonable_actions.get(state, self.actions),
            )
            new_state, effect = world.take_action(state, action)
            reward = self.reward_function(effect)

            # update our map based on what happened
            meaningful_update = self.world_map.update_map(state, action, new_state)

            # learn from what happened
            next_valid_actions = self.world_map.reasonable_actions.get(
                new_state, self.actions
            )
            cur_q = self.q.get((state, action), 0)
            self.q[state, action] = cur_q + self.alpha * (
                reward
                + self.gamma * best_q_value(self.q, new_state, next_valid_actions)
                - cur_q
            )

            # improve from what was learned
            if meaningful_update:
                # if we added a trap or wall a lot of states may change
                for cur_state in self.world_map.world_states:
                    cur_valid_actions = self.world_map.reasonable_actions.get(
                        cur_state, self.actions
                    )
                    self.policy.update(
                        cur_state,
                        get_best_action_from_dict(self.q, cur_state, cur_valid_actions),
                        cur_valid_actions,
                    )
            else:
                # otherwise we only need to worry about the current and next states
                for cur_state in [state, new_state]:
                    cur_valid_actions = self.world_map.reasonable_actions.get(
                        cur_state, self.actions
                    )
                    self.policy.update(
                        cur_state,
                        get_best_action_from_dict(self.q, cur_state, cur_valid_actions),
                        cur_valid_actions,
                    )

            episode_actions.append(action)
            episode_states.append(state)
            episode_rewards.append(reward)

            state = new_state

        return episode_states, episode_rewards, episode_actions
