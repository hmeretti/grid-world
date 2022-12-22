from typing import Final, Collection

from grid_world.action import Action
from grid_world.agents.agent import Agent
from grid_world.agents.policies.epsilon_greedy import EpsilonGreedy
from grid_world.grid_world import GridWorld
from grid_world.state import State
from grid_world.type_aliases import RewardFunction, Q, DecayFunction
from grid_world.utils.policy import (
    sample_action,
    get_best_action_from_dict,
)
from grid_world.utils.returns import returns_from_reward, first_visit_return


class MonteCarloAgent(Agent):
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: Collection[Action] = None,
        gamma: float = 1,
        epsilon: float = 0.1,
        max_steps: int = 10000,
        epsilon_decay: DecayFunction = None,
        q_0: Q = None,
    ):
        """
        Agent implementing a solution based on estimating the value of state-action pairs. Updates are done whenever
        an episode is complete, and only affect visited states.

        :reward_function: the reward function we are trying to maximize
        :actions: actions available to the agent
        :gamma: the gamma discount value to be used when calculating episode returns
        :epsilon: exploration rate to be considered when building policies
        :epsilon_decay: a rule to decay the epsilon parameter.
        :q_0: initial estimates of state-action values, will be considered as a constant 0 if not provided
        """

        self.reward_function: Final = reward_function
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.policy: EpsilonGreedy = EpsilonGreedy(epsilon, actions, epsilon_decay)
        self.gamma = gamma
        self.max_steps = max_steps
        self.q: Q = (
            q_0
            if q_0 is not None
            else {}
        )
        self.u: dict[tuple[State, Action], int] = {}
        self.episode_terminated: bool = False
        self.visited_states: set[State] = set(x for (x, a) in self.q.keys())

        for state in self.visited_states:
            self.policy.update(
                state, get_best_action_from_dict(self.q, state, self.actions)
            )

    def train(self, world: GridWorld, episodes: int = 100) -> tuple[list[int], list[float]]:
        i = 0
        episode_lengths = []
        episode_total_returns = []
        while i < episodes:
            episode_states, episode_rewards = self.run_episode(world)
            if self.episode_terminated:
                episode_returns = returns_from_reward(episode_rewards, self.gamma)
                episode_lengths.append(len(episode_states))
                episode_total_returns.append(episode_returns[0])
                self.policy.decay()
                i += 1

        return episode_lengths, episode_total_returns

    def run_episode(
        self, world: GridWorld, initial_state: State = None
    ) -> tuple[list[float], list[float]]:
        state = initial_state if initial_state is not None else world.initial_state

        self.episode_terminated = False
        episode_states = [state]
        episode_actions = []
        episode_rewards = []

        # run through the world using policy
        for _ in range(self.max_steps):
            action = sample_action(self.policy, state, self.actions)
            state, effect = world.take_action(state, action)
            reward = self.reward_function(effect)

            episode_actions.append(action)
            episode_states.append(state)
            episode_rewards.append(reward)

            if effect == 1:
                self.episode_terminated = True
                break

        # if episode didn't terminate we can't learn anything
        if not self.episode_terminated:
            return [], []

        self._update_visited_states(episode_states)

        # learn from what happened
        episode_returns = returns_from_reward(episode_rewards, self.gamma)
        fvr = first_visit_return(episode_states, episode_actions, episode_returns)
        self._update_q(fvr)

        # improve from what was learned
        for cur_state in self.visited_states:
            self.policy.update(
                cur_state, get_best_action_from_dict(self.q, state, self.actions)
            )

        return episode_states, episode_rewards

    def _update_visited_states(self, episode_states: Collection[State]):
        self.visited_states = self.visited_states.union(set(episode_states))

    def _update_q(self, fvr: dict[tuple[State, Action], float]):
        """
        method to update Q estimates based on the first visit returns.
        This will make changes inplace on the agent Q function

        :param fvr: returns from first visits
        """

        for s, a in fvr:
            self.u[s, a] = self.u.get((s, a), 0) + 1
            self.q[s, a] = self.q.get((s, a), 0) + (fvr[s, a] - self.q.get((s, a), 0)) / self.u.get((s, a), 0)
