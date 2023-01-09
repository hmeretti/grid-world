from typing import Final

from abstractions import Agent, RewardFunction, DecayFunction, Effect, Q
from exploring_agents.grid_world_agents.commons.world_map import WorldMap
from policies import EpsilonExplorer
from grid_world.action import GWorldAction
from grid_world.state import GWorldState
from utils.evaluators import best_q_value
from utils.policy import get_best_action_from_q, sample_action


class QExplorerAgent(Agent):
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: tuple[GWorldAction],
        gamma: float = 1,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: DecayFunction = None,
        alpha_decay: DecayFunction = None,
        q_0: Q = None,
    ):
        """
        Agent very similar to q-learning, that uses an improved exploration policy. The agent uses the same method
        as q-learning to update its Q function, however it uses an exploration policy, more suited to the grid world
        specific problem. It does so by keeping track of a world map, and ignoring actions that are surely bad when
        doing exploration(hitting walls or traps).

        This agent is not well suited for stochastic worlds.

        :param reward_function: the reward function we are trying to maximize
        :param actions: actions available to the agent
        :param gamma: the gamma discount value to be used when calculating episode returns
        :param alpha: learning rate
        :param epsilon: exploration rate to be considered when building policies
        :param epsilon_decay: a rule to decay the epsilon parameter.
        :param alpha_decay: a rule to decay the alpha parameter.
        :param q_0: initial estimates of state-action values, will be considered as a constant 0 if not provided

        """

        self.reward_function: Final = reward_function
        self.policy: EpsilonExplorer = EpsilonExplorer(epsilon, actions, epsilon_decay)
        self.actions: tuple[GWorldAction] = actions
        self.gamma = gamma
        self.alpha = alpha
        self.q: Q = q_0 if q_0 is not None else {}
        self.alpha_decay = alpha_decay if alpha_decay is not None else (lambda x: x)
        self.world_map: WorldMap = WorldMap(
            world_states=set(x for (x, a) in self.q.keys()), actions=self.actions
        )

    def select_action(self, state: GWorldState) -> GWorldAction:
        return sample_action(self.policy, state, self.actions)

    def run_update(
        self,
        state: GWorldState,
        action: GWorldAction,
        effect: Effect,
        next_state: GWorldState,
    ) -> float:
        reward = self.reward_function(effect)

        # update our map based on what happened
        meaningful_update = self.world_map.update_map(state, action, next_state)

        # learn from what happened
        next_valid_actions = self.world_map.reasonable_actions.get(
            next_state, self.actions
        )
        cur_q = self.q.get((state, action), 0)
        self.q[state, action] = cur_q + self.alpha * (
            reward
            + self.gamma * best_q_value(self.q, next_state, next_valid_actions)
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
                    get_best_action_from_q(self.q, cur_state, cur_valid_actions),
                    cur_valid_actions,
                )
        else:
            # otherwise we only need to worry about the current and next states
            for cur_state in [state, next_state]:
                cur_valid_actions = self.world_map.reasonable_actions.get(
                    cur_state, self.actions
                )
                self.policy.update(
                    cur_state,
                    get_best_action_from_q(self.q, cur_state, cur_valid_actions),
                    cur_valid_actions,
                )

        return reward

    def finalize_episode(
        self,
        episode_states: list[GWorldState],
        episode_returns: list[float],
        episode_actions: list[GWorldAction],
    ):
        self.policy.decay()
        self.alpha = self.alpha_decay(self.alpha)
