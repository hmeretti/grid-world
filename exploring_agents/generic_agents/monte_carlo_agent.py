from typing import Final

from abstractions import Agent, RewardFunction, Action, DecayFunction, State, Effect, Q
from exploring_agents.policies.epsilon_greedy import EpsilonGreedy
from utils.policy import get_best_action_from_q, sample_action
from utils.returns import first_visit_return


class MonteCarloAgent(Agent):
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: list[Action],
        gamma: float = 1,
        epsilon: float = 0.1,
        epsilon_decay: DecayFunction = None,
        max_steps: int = 10000,
        q_0: Q = None,
    ):
        """
        Agent implementing a solution based on estimating the value of state-action pairs. Updates are done whenever
        an episode is complete, and only affect visited states.

        :param reward_function: the reward function we are trying to maximize
        :param actions: actions available to the agent
        :param gamma: the gamma discount value to be used when calculating episode returns
        :param epsilon: exploration rate to be considered when building policies
        :param epsilon_decay: a rule to decay the epsilon parameter.
        :param q_0: initial estimates of state-action values, will be considered as a constant 0 if not provided
        """

        self.reward_function: Final = reward_function
        self.actions: Final = actions
        self.policy: EpsilonGreedy = EpsilonGreedy(epsilon, actions, epsilon_decay)
        self.gamma = gamma
        self.max_steps = max_steps
        self.q: Q = q_0 if q_0 is not None else {}
        self.u: dict[tuple[State, Action], int] = {}
        self.episode_terminated: bool = False
        self.visited_states: set[State] = set(x for (x, a) in self.q.keys())

        for state in self.visited_states:
            self.policy.update(
                state, get_best_action_from_q(self.q, state, self.actions)
            )

    def select_action(self, state: State) -> Action:
        return sample_action(self.policy, state, self.actions)

    def run_update(
        self, state: State, action: Action, effect: Effect, next_state: State
    ) -> float:
        reward = self.reward_function(effect)
        self.visited_states.add(next_state)

        # learn from what happened
        return reward

    def finalize_episode(
        self,
        episode_states: list[State],
        episode_returns: list[float],
        episode_actions: list[Action],
    ):
        # learn from what happened
        fvr = first_visit_return(episode_states, episode_actions, episode_returns)
        self._update_q(fvr)

        # improve from what we learned
        for cur_state in self.visited_states:
            self.policy.update(
                cur_state, get_best_action_from_q(self.q, cur_state, self.actions)
            )

        self.policy.decay()

    def _update_q(self, fvr: dict[tuple[State, Action], float]):
        """
        method to update Q estimates based on the first visit returns.
        This will make changes inplace on the agent Q function

        :param fvr: returns from first visits
        """

        for s, a in fvr:
            self.u[s, a] = self.u.get((s, a), 0) + 1
            self.q[s, a] = self.q.get((s, a), 0) + (
                fvr[s, a] - self.q.get((s, a), 0)
            ) / self.u.get((s, a), 0)
