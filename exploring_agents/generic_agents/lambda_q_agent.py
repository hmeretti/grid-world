from typing import Final

from abstractions import Agent, RewardFunction, Action, DecayFunction, State, Effect, Q
from exploring_agents.commons.eligibility_trace import EligibilityTrace
from policies import EpsilonGreedy
from utils.evaluators import best_q_value
from utils.policy import (
    get_best_action_from_q,
    sample_action_and_exploration,
    sample_action,
)


class LambdaQAgent(Agent):
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: tuple[Action, ...],
        gamma: float = 1,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        et_lambda: float = 0.5,
        et_kind: str = "accumulating",
        epsilon_decay: DecayFunction = None,
        alpha_decay: DecayFunction = None,
        lambda_decay: DecayFunction = None,
        q_0: Q = None,
    ):
        """
        Agent similar to q-learning. Uses eligibility traces to update more state action
        pairs than the current one.

        :param reward_function: the reward function we are trying to maximize
        :param actions: actions available to the agent
        :param gamma: the gamma discount value to be used when calculating episode returns
        :param alpha: learning rate
        :param epsilon: exploration rate to be considered when building policies
        :param epsilon_decay: a rule to decay the epsilon parameter.
        :param alpha_decay: a rule to decay the alpha parameter.
        :param lambda_decay: a rule to decay the lambda parameter.
        :param q_0: initial estimates of state-action values, will be considered as a constant 0 if not provided
        """

        self.reward_function: Final = reward_function
        self.policy: EpsilonGreedy = EpsilonGreedy(epsilon, actions, epsilon_decay)
        self.actions: Final = actions
        self.gamma = gamma
        self.alpha = alpha
        self.et_lambda = et_lambda
        self.et_kind = et_kind
        self.q: Q = q_0 if q_0 is not None else {}
        self.alpha_decay = alpha_decay if alpha_decay is not None else (lambda x: x)
        self.lambda_decay = lambda_decay if lambda_decay is not None else (lambda x: x)
        self.visited_states: set[State] = set(x for (x, a) in self.q.keys())
        self.next_action = None
        self.eligibility_trace = EligibilityTrace(
            et_lambda=self.et_lambda, gamma=self.gamma, kind=self.et_kind, alpha=alpha
        )

        for state in self.visited_states:
            self.policy.update(
                state, get_best_action_from_q(self.q, state, self.actions)
            )

    # overriding the method
    def select_action(self, state: State, use_cached_action: bool = True) -> Action:
        """
        selects an action from a state based on the agent policy
        during training we select the next action to be taken whenever we run an update
        we can bypass this behaviour with use_cached_action

        :param state: the state to select the action from
        :param use_cached_action: flag indicating whether we should use pre-selected action
        :return: the selected action
        """

        return (
            self.next_action
            if self.next_action is not None and use_cached_action
            else sample_action(self.policy, state, self.actions)
        )

    def run_update(
        self, state: State, action: Action, effect: Effect, next_state: State
    ) -> float:
        self.eligibility_trace.visited_arguments_update(state, action)
        reward = self.reward_function(effect)

        # learn from what happened
        delta = (
            reward
            + self.gamma * best_q_value(self.q, next_state, self.actions)
            - self.q.get((state, action), 0)
        )
        update_dict = {
            sap: self.q.get(sap, 0) + self.alpha * delta * self.eligibility_trace(*sap)
            for sap in self.eligibility_trace.get_relevant_arguments()
        }
        self.q.update(update_dict)

        # improve from what was learned
        for cur_state in {state for state, actiongi in update_dict.keys()}:
            self.policy.update(
                cur_state, get_best_action_from_q(self.q, cur_state, self.actions)
            )

        # update traces: look ahead to see if we will explore
        self.next_action, has_explored = sample_action_and_exploration(
            self.policy, next_state, self.actions
        )
        if has_explored:
            # if so we reset everything
            self.eligibility_trace.reset()
        else:
            # we just update the traces
            self.eligibility_trace.update()

        return reward

    def finalize_episode(
        self,
        episode_states: list[State],
        episode_returns: list[float],
        episode_actions: list[Action],
    ):

        self.policy.decay()
        self.alpha = self.alpha_decay(self.alpha)
        self.et_lambda = self.lambda_decay(self.et_lambda)
        self.eligibility_trace.reset()
        self.eligibility_trace.et_lambda = self.et_lambda
