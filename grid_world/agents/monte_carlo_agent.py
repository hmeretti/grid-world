from typing import Final, Collection

from grid_world.action import Action
from grid_world.grid_world import GridWorld
from grid_world.state import State
from grid_world.type_aliases import Police, RewardFunction, Q
from grid_world.utils.police import (
    get_random_police,
    sample_action,
    get_e_greedy_police,
)
from grid_world.utils.returns import returns_from_reward, first_visit_return


class MonteCarloAgent:
    def __init__(
        self,
        world: GridWorld,
        reward_function: RewardFunction,
        actions: Collection[Action] = None,
        police: Police = None,
        gamma: float = 1,
        epsilon: float = 0.1,
        q_0: Q = None,
    ):
        self.world: Final = world
        self.reward_function: Final = reward_function
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.police = Police if police is not None else get_random_police(self.actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: Q = (
            q_0
            if q_0 is not None
            else {(s, a): 0 for s in self.world.states for a in self.actions}
        )
        self.u: dict[tuple[State, Action], int] = {
            (s, a): 0 for s in self.world.states for a in self.actions
        }
        self.visited_states: set[State] = set()

    def train(
        self, episodes: int = 100, max_steps: int = 10000
    ) -> tuple[list[int], list[float]]:
        i = 0
        episode_lengths = []
        episode_returns = []
        while i < episodes:
            episode_terminated, episode_length, episode_return = self.run_episode(
                max_steps=max_steps
            )
            if episode_terminated:
                episode_lengths.append(episode_length)
                episode_returns.append(episode_return)
                i += 1

        return episode_lengths, episode_returns

    def run_episode(
        self, initial_state: State = None, max_steps: int = 10000
    ) -> [bool, int]:
        state = initial_state if initial_state is not None else self.world.initial_state

        episode_terminated = False
        episode_states = [state]
        episode_actions = []
        episode_rewards = []

        # run through the world using police
        for _ in range(max_steps):
            action = sample_action(self.police, state, self.actions)
            state, effect = self.world.take_action(state, action)
            reward = self.reward_function(effect)

            episode_actions.append(action)
            episode_states.append(state)
            episode_rewards.append(reward)

            if state.kind == "terminal":
                episode_terminated = True
                break

        # if episode didn't terminate we can't learn anything
        if not episode_terminated:
            return episode_terminated, 0, 0

        self._update_visited_states(episode_states)

        # learn from what happened
        episode_returns = returns_from_reward(episode_rewards, self.gamma)
        fvr = first_visit_return(episode_states, episode_actions, episode_returns)
        self._update_q(fvr)

        # improve from what was learned
        self.police = get_e_greedy_police(
            self.q, self.visited_states, self.actions, self.epsilon
        )

        return episode_terminated, len(episode_states), episode_returns[0]

    def _update_visited_states(self, episode_states: Collection[State]):
        self.visited_states = self.visited_states.union(set(episode_states))

    def _update_q(self, fvr: dict[tuple[State, Action], float]):
        """
        method to update Q estimates based on the first visit returns.
        This will make changes inplace on the agent Q function

        :param fvr: returns from first visits
        """

        for s, a in fvr:
            self.u[s, a] = self.u[s, a] + 1
            self.q[s, a] = (
                self.q[s, a] + (fvr[s, a] - self.q[s, a]) / self.u[s, a]
            )
