from typing import Final, Collection

from grid_world.action import Action
from grid_world.grid_world import GridWorld
from grid_world.state import State
from grid_world.type_aliases import Police, RewardFunction, Q
from grid_world.utils.evaluators import best_q_value
from grid_world.utils.police import (
    get_random_police,
    sample_action,
    get_explorer_police,
    get_reasonable_actions,
)
from grid_world.utils.returns import returns_from_reward
from utils.operations import add_tuples


class QExplorerAgent:
    def __init__(
        self,
        reward_function: RewardFunction,
        actions: Collection[Action] = None,
        police: Police = None,
        gamma: float = 1,
        alpha: float = 0.1,
        epsilon: float = 0.1,
    ):
        self.reward_function: Final = reward_function
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.police = Police if police is not None else get_random_police(self.actions)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q: Q = dict()
        self.world_map: set[State] = set()

    def train(
        self,
        world: GridWorld,
        episodes: int = 100,
    ) -> tuple[list[int], list[float]]:
        episode_lengths = []
        episode_total_returns = []
        for _ in range(episodes):
            episode_actions, episode_states, episode_rewards = self.run_episode(world)
            episode_returns = returns_from_reward(episode_rewards, self.gamma)
            episode_lengths.append(len(episode_actions))
            episode_total_returns.append(episode_returns[0])

        return episode_lengths, episode_total_returns

    def run_episode(self, world: GridWorld) -> [bool, int]:
        state = world.initial_state

        episode_states = [state]
        episode_actions = []
        episode_rewards = []

        # run through the world while updating q the police and our map as we go
        while state.kind != "terminal":
            action = sample_action(self.police, state, self.actions)
            new_state, effect = world.take_action(state, action)
            reward = self.reward_function(effect)
            # if we hit a wall or the border the agent will stand still, in this case we mark it as a wall
            if new_state == state:
                self.world_map.add(
                    State(add_tuples(state.coordinates, action.direction), "wall")
                )
            else:
                self.world_map.add(new_state)

            # determine reasonable actions
            reasonable_actions = {
                s: get_reasonable_actions(self.world_map, s, self.actions)
                for s in self.world_map
            }

            # learn from what happened
            cur_q = self.q.get((state, action), 0)
            self.q[state, action] = cur_q + self.alpha * (
                reward
                + self.gamma
                * best_q_value(
                    self.q, new_state, reasonable_actions.get(new_state, self.actions)
                )
                - cur_q
            )

            # improve from what was learned
            self.police = get_explorer_police(
                self.q, self.world_map, self.actions, reasonable_actions, self.epsilon
            )

            state = new_state
            episode_actions.append(action)
            episode_states.append(state)
            episode_rewards.append(reward)

        return episode_actions, episode_states, episode_rewards
