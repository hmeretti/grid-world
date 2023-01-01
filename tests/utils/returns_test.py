import numpy as np

from tests.constants.actions import a0, a1
from tests.constants.states import s0, s1
from utils.returns import returns_from_reward, first_visit_return


class TestReturns:
    @staticmethod
    def test_returns_from_reward():
        rewards = [1, 2, -2, 1]
        returns0 = returns_from_reward(rewards, gamma=1)
        assert returns0 == [2, 1, -1, 1]

        gamma = 0.9
        returns1 = returns_from_reward(rewards, gamma=gamma)
        expected_returns = [
            1 + gamma * (2 + gamma * (-2 + gamma * 1)),
            2 + gamma * (-2 + gamma * 1),
            -2 + gamma * 1,
            1,
        ]
        assert all([np.isclose(x, y) for x, y in zip(returns1, expected_returns)])

    @staticmethod
    def test_first_visit_returns():
        returns = [1, 2, -2, 1]
        states = [s0, s1, s0, s1]
        actions = [a0, a1, a0, a0]
        fvr = first_visit_return(states, actions, returns)
        assert fvr[s0, a0] == 1
        assert fvr[s1, a1] == 2
        assert fvr[s1, a0] == 1
        assert len(fvr) == 3
