from typing import Final, Collection

from grid_world.action import Action
from grid_world.agents.policies.epsilon_greedy import EpsilonGreedy
from grid_world.grid_world import GridWorld
from grid_world.state import State
from grid_world.type_aliases import RewardFunction, Q, DecayFunction
from grid_world.utils.policy import (
    sample_action,
    get_best_action_from_dict,
)
from grid_world.utils.returns import returns_from_reward
from grid_world.agents.commons.eligibility_trace import EligibilityTrace


class LambdaSarsaAgent:
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: Collection[Action] = None,
        gamma: float = 1,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        et_lambda: float = 0.5,
        et_kind: str = "accumulating",
        epsilon_decay: DecayFunction = None,
        alpha_decay: DecayFunction = None,
        q_0: Q = None,
    ):
        """
        Agent implementing a solution based on estimating the value of state-action pairs. It updates values after
        every action, by observing results, selecting a next action and bootstrapping values from this information.

        :reward_function: the reward function we are trying to maximize
        :actions: actions available to the agent
        :gamma: the gamma discount value to be used when calculating episode returns
        :alpha: learning rate
        :epsilon: exploration rate to be considered when building policies
        :epsilon_decay: a rule to decay the epsilon parameter. Should be a function of epsilon,
            that returns another function, which will determine for each epoch the value of epsilon
        :alpha_decay: a rule to decay the alpha parameter. Should be a function of epsilon,
            that returns another function, which will determine for each epoch the value of epsilon
        :q_0: initial estimates of state-action values, will be considered as a constant 0 if not provided

        """
        self.reward_function: Final = reward_function
        self.policy: EpsilonGreedy = EpsilonGreedy(epsilon, actions, epsilon_decay)
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.gamma = gamma
        self.alpha = alpha
        self.et_lambda = et_lambda
        self.et_kind = et_kind
        self.q: Q = q_0 if q_0 is not None else {}
        self.alpha_decay = alpha_decay if alpha_decay is not None else (lambda x: x)
        self.visited_states: set[State] = set(x for (x, a) in self.q.keys())

        for state in self.visited_states:
            self.policy.update(
                state, get_best_action_from_dict(self.q, state, self.actions)
            )

    def train(
        self,
        world: GridWorld,
        episodes: int = 100,
    ) -> tuple[list[int], list[float]]:
        episode_lengths = []
        episode_total_returns = []
        for episode in range(episodes):
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
        eligibility_trace = EligibilityTrace(
            et_lambda=self.et_lambda, gamma=self.gamma, kind=self.et_kind
        )
        state = initial_state if initial_state is not None else world.initial_state
        action = sample_action(self.policy, state, self.actions)

        episode_states = []
        episode_rewards = []
        episode_actions = []

        # run through the world while updating q, eligibility trace, and the policy as we go
        effect = 0
        while effect != 1:
            eligibility_trace.update(state, action)
            new_state, effect = world.take_action(state, action)
            next_action = sample_action(self.policy, new_state, self.actions)
            reward = self.reward_function(effect)
            self.visited_states.add(new_state)

            # learn from what happened
            delta = (
                reward
                + self.gamma * self.q.get((new_state, next_action), 0)
                - self.q.get((state, action), 0)
            )
            update_dict = {
                sap: self.q.get(sap, 0) + self.alpha * delta * eligibility_trace(*sap)
                for sap in eligibility_trace.get_relevant_state_actions()
            }
            self.q.update(update_dict)

            # improve from what was learned
            self.policy.update(
                state, get_best_action_from_dict(self.q, state, self.actions)
            )

            # for iter_state in {x for x, _ in update_dict}:
            #     self.policy.update(
            #         iter_state, get_best_action_from_dict(self.q, iter_state, self.actions)
            #     )

            episode_states.append(state)
            episode_rewards.append(reward)
            episode_actions.append(action)

            state = new_state
            action = next_action

        return episode_states, episode_rewards, episode_actions
