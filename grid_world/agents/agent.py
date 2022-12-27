from abstractions import State, Action, World


class Agent:
    """
    Abstract policy class. Concrete extensions should implement

    :__call__: the function that tells for each state, the probability of an action
    :update: a function to update the policy

    """

    def train(
        self,
        world: World,
        episodes: int,
    ) -> tuple[list[int], list[float]]:
        raise NotImplementedError("train method not implemented")

    def run_episode(
        self, world: World, initial_state: State
    ) -> tuple[list[State], list[float], list[Action]]:
        raise NotImplementedError("run_episode method not implemented")
