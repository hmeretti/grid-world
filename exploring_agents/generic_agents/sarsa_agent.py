from typing import Final

from abstractions import Agent, RewardFunction, Action, DecayFunction, State, Effect, Q
from exploring_agents.policies.epsilon_greedy import EpsilonGreedy
from utils.policy import get_best_action_from_q, sample_action


class SarsaAgent(Agent):
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: list[Action],
        gamma: float = 1,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: DecayFunction = None,
        alpha_decay: DecayFunction = None,
        q_0: Q = None,
    ):
        """
        Agent implementing a solution based on estimating the value of state-action pairs. It updates values after
        every action, by observing results and bootstrapping values from what is expected to be the best policy
        for the following state.

        :reward_function: the reward function we are trying to maximize
        :actions: actions available to the agent
        :gamma: the gamma discount value to be used when calculating episode returns
        :alpha: learning rate
        :epsilon: exploration rate to be considered when building policies
        :epsilon_decay: a rule to decay the epsilon parameter.
        :alpha_decay: a rule to decay the alpha parameter.
        :q_0: initial estimates of state-action values, will be considered as a constant 0 if not provided
        """

        self.reward_function: Final = reward_function
        self.policy: EpsilonGreedy = EpsilonGreedy(epsilon, actions, epsilon_decay)
        self.actions: Final = actions
        self.gamma = gamma
        self.alpha = alpha
        self.q: Q = q_0 if q_0 is not None else {}
        self.alpha_decay = alpha_decay if alpha_decay is not None else (lambda x: x)
        self.visited_states: set[State] = set(x for (x, a) in self.q.keys())
        self.next_action: [None, Action] = None

        for state in self.visited_states:
            self.policy.update(
                state, get_best_action_from_q(self.q, state, self.actions)
            )

    # overriding the method
    def select_action(self, state: State) -> Action:
        return (
            self.next_action
            if self.next_action is not None
            else sample_action(self.policy, state, self.actions)
        )

    def run_update(
        self, state: State, action: Action, effect: Effect, next_state: State
    ) -> float:
        self.next_action = sample_action(self.policy, next_state, self.actions)
        reward = self.reward_function(effect)
        self.visited_states.add(next_state)

        # learn from what happened
        self.q[state, action] = self.q.get((state, action), 0) + self.alpha * (
            reward
            + self.gamma * self.q.get((next_state, self.next_action), 0)
            - self.q.get((state, action), 0)
        )

        # improve from what was learned
        self.policy.update(state, get_best_action_from_q(self.q, state, self.actions))

        return reward

    def finalize_episode(
        self,
        episode_states: list[State],
        episode_returns: list[float],
        episode_actions: list[Action],
    ):
        self.next_action = None
        self.policy.decay()
        self.alpha = self.alpha_decay(self.alpha)
