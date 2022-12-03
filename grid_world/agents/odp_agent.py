from typing import Final, Collection

from dynamic_programing.policy_improvement import dynamic_programing_gpi
from grid_world.action import Action
from grid_world.agents.world_map import WorldMap
from grid_world.grid_world import GridWorld
from grid_world.type_aliases import Policy, RewardFunction
from grid_world.utils.policy import (
    get_random_policy,
    sample_action,
)
from grid_world.utils.returns import returns_from_reward


class ODPAgent:
    def __init__(
        self,
        reward_function: RewardFunction,
        world_height: int,
        world_length: int,
        actions: Collection[Action] = None,
        gamma: float = 1,
    ):
        self.reward_function: Final = reward_function
        self.actions: Final = actions if actions is not None else tuple(Action)
        self.gamma = gamma
        self.world_map: WorldMap = WorldMap(world_states=set(), actions=self.actions)
        self.final_state_known = False
        self.optimal_path_found = False
        self.policy = get_random_policy(self.actions)
        self.world_length = world_length
        self.world_height = world_height

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
        perfect_run = True

        episode_states = [state]
        episode_actions = []
        episode_rewards = []

        # run through the world while updating q the policy and our map as we go
        while state.kind != "terminal":
            action = sample_action(self.policy, state, self.actions)
            new_state, effect = world.take_action(state, action)
            reward = self.reward_function(effect)

            # update our map based on what happened
            self.world_map.update_map(state, action, new_state)

            # in case we already know the final state, and we hit a wall or trap we need to update the policy
            if self.final_state_known and (
                new_state == state or new_state.kind == "trap"
            ):
                self.policy = self._build_odp_policy()
                perfect_run = False

            state = new_state
            episode_actions.append(action)
            episode_states.append(state)
            episode_rewards.append(reward)

        # in case this wasn't a randon run, and we did not have to make path corrections we have found an optimal path
        self.optimal_path_found = self.final_state_known and perfect_run

        # this triggers after the first successful run
        if not self.final_state_known:
            self.policy = self._build_odp_policy()
            self.final_state_known = True

        return episode_actions, episode_states, episode_rewards

    def _build_opt_world(self, world_length: int, world_height: int) -> GridWorld:
        return GridWorld(
            grid_shape=(world_length, world_height),
            terminal_states_coordinates=self._get_state_by_kind(
                "terminal", world_length, world_height
            ),
            walls_coordinates=self._get_state_by_kind(
                "wall", world_length, world_height
            ),
            traps_coordinates=self._get_state_by_kind(
                "trap", world_length, world_height
            ),
        )

    def _get_state_by_kind(self, kind, world_length, world_height):
        return tuple(
            a.coordinates
            for a in self.world_map.world_states
            if (
                a.kind == kind
                and (0 <= a.coordinates[0] < world_length)
                and (0 <= a.coordinates[1] < world_height)
            )
        )

    @staticmethod
    def _get_world_model(world):
        return lambda s, a: lambda x: 1 if x == world.take_action(s, a)[0] else 0

    def _build_odp_policy(self) -> Policy:
        world = self._build_opt_world(self.world_length, self.world_height)
        world_model = self._get_world_model(world)

        rewards_dict = {
            (s, a): self.reward_function(world.take_action(s, a)[1])
            for s in world.states
            for a in self.actions
        }

        policy, _ = dynamic_programing_gpi(
            world_model=world_model,
            reward_function=lambda x, y: rewards_dict[(x, y)],
            actions=self.actions,
            states=world.states,
        )
        return policy
