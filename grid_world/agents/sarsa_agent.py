from typing import Final, Collection, Callable

from grid_world.action import Action
from grid_world.grid_world import GridWorld
from grid_world.state import State
from grid_world.type_aliases import Policy, RewardFunction, Q
from grid_world.utils.policy import (
    get_random_policy,
    sample_action,
    get_e_greedy_policy, get_best_action_from_q,
)
from grid_world.utils.returns import returns_from_reward


class SarsaAgent:
    def __init__(
        self,
        world: GridWorld,
        reward_function: RewardFunction,
        actions: Collection[Action] = None,
        gamma: float = 1,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: Callable[[float], Callable[[float], float]] = lambda x: lambda y: x,
        alpha_decay: Callable[[float], Callable[[float], float]] = lambda x: lambda y: x,
        q_0: Q = None,
    ):
        """
        Agent implementing a solution based on estimating the value of state-action pairs. It updates values after
        every action, by observing results, selecting a next action and bootstrapping values from this information.

        :world: the world this agent will explore
        :reward_function: the reward function we are trying to maximize
        :actions: actions available to the agent
        :gamma: the gamma discount value to be used when calculating episode returns
        :alpha: learning rate
        :epsilon: exploration rate to be considered when building policies
        :epsilon_decay: a rule to decay the epsilon parameter. Should be a function of epsilon, that returns another function, which will determine for each epoch the value of epsilon
        :alpha_decay: a rule to decay the alpha parameter. Should be a function of epsilon, that returns another function, which will determine for each epoch the value of epsilon
        :q_0: initial estimates of state-action values, will be considered as a constant 0 if not provided

        """
        self.world: Final = world
        self.reward_function: Final = reward_function
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay(epsilon)
        self.alpha_decay = alpha_decay(alpha)
        self.q: Q = (
            q_0
            if q_0 is not None
            else {(s, a): 0 for s in self.world.states for a in self.actions}
        )
        self.visited_states: set[State] = set(x for (x, a) in self.q.keys())
        self.policy_map: dict[[State, Action], float] = {}

        for x in self.visited_states:
            self._update_policy_dict(x)

    def policy(self, state: State, action: Action) -> float:
        return self.policy_map.get((state, action), 1/len(self.actions))

    def _update_policy_dict(self, state: State) -> None:
        best_action = get_best_action_from_q(self.q, state, self.actions)
        for cur_a in self.actions:
            self.policy_map[state, cur_a] = 1-self.epsilon if cur_a == best_action else self.epsilon/(len(self.actions) - 1)

    def train(
        self,
        episodes: int = 100,
    ) -> tuple[list[int], list[float]]:
        episode_lengths = []
        episode_total_returns = []
        for episode in range(episodes):
            episode_actions, episode_states, episode_rewards = self.run_episode()
            episode_returns = returns_from_reward(episode_rewards, self.gamma)
            episode_lengths.append(len(episode_actions))
            episode_total_returns.append(episode_returns[0])
            self.epsilon = self.epsilon_decay(episode)
            self.alpha = self.alpha_decay(episode)

        return episode_lengths, episode_total_returns

    def run_episode(self, initial_state: State = None) -> [bool, int]:
        state = initial_state if initial_state is not None else self.world.initial_state

        episode_states = [state]
        episode_actions = []
        episode_rewards = []

        # run through the world while updating q and the policy as we go
        action = sample_action(self.policy, state, self.actions)
        effect = 0
        while effect != 1:
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
            self._update_policy_dict(state)

            state = new_state
            episode_actions.append(action)
            episode_states.append(state)
            episode_rewards.append(reward)
            action = next_action

        return episode_actions, episode_states, episode_rewards
