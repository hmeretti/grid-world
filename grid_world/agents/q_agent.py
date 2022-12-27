from typing import Final, Collection

from grid_world.action import GWorldAction
from grid_world.agents.agent import Agent
from grid_world.agents.policies.epsilon_greedy import EpsilonGreedy
from grid_world.grid_world import GridWorld
from grid_world.state import GWorldState
from abstractions import RewardFunction, Q, DecayFunction
from utils.evaluators import best_q_value
from utils.policy import (
    sample_action,
    get_best_action_from_dict,
)
from utils.returns import returns_from_reward


class QAgent(Agent):
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: Collection[GWorldAction] = tuple(GWorldAction),
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
        self.visited_states: set[GWorldState] = set(x for (x, a) in self.q.keys())

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
        self, world: GridWorld, initial_state: GWorldState = None
    ) -> tuple[list[GWorldState], list[float], list[GWorldAction]]:
        state = initial_state if initial_state is not None else world.initial_state

        episode_states = []
        episode_rewards = []
        episode_actions = []

        # run through the world while updating q and the policy as we go
        effect = 0
        while effect != 1:
            action = sample_action(self.policy, state, self.actions)
            new_state, effect = world.take_action(state, action)
            reward = self.reward_function(effect)
            self.visited_states.add(new_state)

            # learn from what happened
            self.q[state, action] = self.q.get((state, action), 0) + self.alpha * (
                reward
                + self.gamma * best_q_value(self.q, new_state, self.actions)
                - self.q.get((state, action), 0)
            )

            # improve from what was learned
            self.policy.update(
                state, get_best_action_from_dict(self.q, state, self.actions)
            )

            episode_states.append(state)
            episode_rewards.append(reward)
            episode_actions.append(action)

            state = new_state

        return episode_states, episode_rewards, episode_actions
