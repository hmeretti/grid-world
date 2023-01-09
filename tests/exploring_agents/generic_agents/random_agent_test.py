import numpy as np

from exploring_agents import RandomAgent
from notebooks.utils.basics import basic_reward
from tests.constants.actions import a0, a1, a2, a3
from tests.constants.states import s0, s1


class TestRandomAgent:
    @staticmethod
    def test_select_action():
        # This is a stochastic test, which is subject to random failure, although very unlikely

        agent1 = RandomAgent(
            reward_function=basic_reward, actions=(a0, a1, a2), gamma=1
        )
        iterations = 1e5
        actions = [agent1.select_action(s0) for _ in range(int(iterations))]
        assert np.isclose(
            np.sum([x == a0 for x in actions]) / iterations, 1 / 3, atol=1e-2
        )
        assert np.isclose(
            np.sum([x == a1 for x in actions]) / iterations, 1 / 3, atol=1e-2
        )
        assert np.isclose(
            np.sum([x == a2 for x in actions]) / iterations, 1 / 3, atol=1e-2
        )
        assert np.sum([x == a3 for x in actions]) == 0

        actions1 = [agent1.select_action(s1) for _ in range(int(iterations))]
        assert np.isclose(
            np.sum([x == a0 for x in actions1]) / iterations, 1 / 3, atol=1e-2
        )
        assert np.isclose(
            np.sum([x == a1 for x in actions1]) / iterations, 1 / 3, atol=1e-2
        )
        assert np.isclose(
            np.sum([x == a2 for x in actions1]) / iterations, 1 / 3, atol=1e-2
        )
