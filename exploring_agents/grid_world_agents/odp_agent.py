from typing import Final

from abstractions import Agent, RewardFunction, Action, State, Effect, Policy
from dynamic_programing.policy_improvement import dynamic_programing_gpi
from exploring_agents.grid_world_agents.commons.world_map import WorldMap
from grid_world.action import GWorldAction
from grid_world.grid_world import GridWorld
from grid_world.state import GWorldState
from utils.policy import get_random_policy


class ODPAgent(Agent):
    def __init__(
            self,
            reward_function: RewardFunction,
            world_shape: tuple[int, int],
            actions: list[GWorldAction],
            terminal_coordinates: tuple[int, int] = None,
            gamma: float = 1,
    ):
        """
        Agent implementing a solution based on dynamic programing.

        This agent goes through the world, keeping a map, and "planning" its routes using DP. It starts with a
        random policy, that will eventually lead it to the goal. Once it has found the goal, it builds an "optimistic"
        model of the world(one where every unexplored state is empty, which for our problem is the case that would lead
        to the best possible solution) and uses DP to find the best policy in such a world. It repeats this
        process whenever the optimistic assumption of the world leads to a mistake(hitting a wall or a trap).

        We treat the agent as if it doesn't know anything about the world it will explore. For this reason  dimensions
        for the optimistic world should be provided. To guarantee convergence to an optimal solution these should be
        as big as the actual world it will explore(or large enough to contain the optimal path). Making the optimistic
        world bigger shouldn't affect the final solution, but can make exploration longer, and will make the DP gpi
        algorithm slower.

        :reward_function: the reward function we are trying to maximize
        :world_height: height of the optimistic world we will consider
        :world_length: length of the optimistic world we will consider
        :actions: actions available to the agent
        :gamma: the gamma discount value to be used when calculating episode returns
        """
        self.reward_function: Final = reward_function
        self.actions: Final[list[GWorldAction]] = actions
        self.gamma = gamma
        self.world_map: WorldMap = (
            WorldMap(world_states=set(), actions=self.actions)
            if terminal_coordinates is None
            else WorldMap(
                world_states={GWorldState(terminal_coordinates, "terminal")},
                actions=self.actions,
            )
        )
        self.final_state_known = terminal_coordinates is not None
        self.optimal_path_found = False
        self.world_shape = world_shape
        self.perfect_run = True
        self.policy = (
            self._build_odp_policy(False)
            if self.final_state_known
            else get_random_policy(self.actions)
        )

    def run_update(
        self, state: GWorldState, action: GWorldAction, effect: Effect, next_state: GWorldState
    ) -> float:
        reward = self.reward_function(effect)

        # update our map based on what happened
        meaningful_update = self.world_map.update_map(state, action, next_state)

        # in case we already know the final state, and we hit a wall or trap we need to update the policy
        if self.final_state_known and meaningful_update:
            self.policy = self._build_odp_policy()
            self.perfect_run = False

        return reward

    def finalize_episode(
            self,
            episode_states: list[State],
            episode_returns: list[float],
            episode_actions: list[Action],
    ):
        # in case this wasn't a randon run, and we did not have to make path corrections we have found an optimal path
        self.optimal_path_found = self.final_state_known and self.perfect_run
        # reset flag for next run
        self.perfect_run = True

        # this triggers after the first successful run, if we didn't know the final state
        if not self.final_state_known:
            self.policy = self._build_odp_policy()
            self.final_state_known = True

    def build_opt_world(self) -> GridWorld:
        return GridWorld(
            grid_shape=self.world_shape,
            terminal_states_coordinates=self._get_state_by_kind(
                "terminal",
            ),
            walls_coordinates=self._get_state_by_kind(
                "wall",
            ),
            traps_coordinates=self._get_state_by_kind(
                "trap",
            ),
        )

    def _get_state_by_kind(self, kind):
        return tuple(
            a.coordinates
            for a in self.world_map.world_states
            if (
                a.kind == kind
                and (0 <= a.coordinates[0] < self.world_shape[0])
                and (0 <= a.coordinates[1] < self.world_shape[1])
            )
        )

    @staticmethod
    def _get_world_model(world):
        return lambda s, a: lambda x: 1 if x == world.take_action(s, a)[0] else 0

    def _build_odp_policy(self, verbose: bool = False) -> Policy:
        world = self.build_opt_world()
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
            verbose=verbose,
        )
        return policy
