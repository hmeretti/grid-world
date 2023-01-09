from abc import ABC, abstractmethod
from typing import Generic

import dill

from abstractions import Effect, Policy
from abstractions.type_vars import ActionTypeVar, StateTypeVar


class Agent(ABC, Generic[ActionTypeVar, StateTypeVar]):
    """
    Abstract agent class defining interface to be used by training
    process.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("init not implemented")

    gamma: float = NotImplemented
    actions: tuple[ActionTypeVar, ...] = NotImplemented
    policy: Policy = NotImplemented

    @abstractmethod
    def select_action(self, state: StateTypeVar) -> ActionTypeVar:
        """
        selects an action from a state based on the agent policy

        :param state: the state to select the action from
        :return: the selected action
        """
        raise NotImplementedError("select_action method not implemented")

    @abstractmethod
    def run_update(
        self,
        state: StateTypeVar,
        action: ActionTypeVar,
        effect: Effect,
        next_state: StateTypeVar,
    ) -> float:
        """
        updates agent policy for a "state action effect state" sequence,
        uses the effect to determine a reward, which is returned

        :param state: the state where the agent chose an action
        :param action: the chosen action
        :param effect: the effect sent by the training loop
        :param next_state: next state for which the agent will need to choose an action
        :return: the reward determined by the agent
        """
        raise NotImplementedError("run_update method not implemented")

    @abstractmethod
    def finalize_episode(
        self,
        episode_states: list[StateTypeVar],
        episode_returns: list[float],
        episode_actions: list[ActionTypeVar],
    ):
        """
        Runs any updates the agent may wish to do by the end of an episode

        :param episode_states: the states visited by the agent during the episode
        :param episode_returns: the returns the agent got during the episode
        :param episode_actions: the actions chosen by the agent during the episode
        """
        raise NotImplementedError("finalize_episode method not implemented")

    def dump(self, filename: str) -> None:
        """
        creates an artifact of the agent for later use.

        :param filename: the path where the file will be saved
        """
        with open(f"{filename}.agent", "wb") as handle:
            dill.dump(self, handle)

    @staticmethod
    def load(filename: str):
        """
        loads an artifact of an agent

        :param filename: the path to the file containing the agent
        :return: a loaded agent
        """
        with open(f"{filename}.agent", "rb") as handle:
            return dill.load(handle)
