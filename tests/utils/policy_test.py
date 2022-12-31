import numpy as np

from tests.constants.actions import a0, a1, a2, a3
from tests.constants.states import s0, s1, s2, s3
from utils.policy import get_best_action_from_q, get_random_policy

states0 = [s0, s1, s2]
actions0 = [a0, a1, a2]


class TestPolicy:
    @staticmethod
    def test_get_best_action_from_q(q_0, q_1, q_2):
        assert get_best_action_from_q(q_0, s0, actions0) == a0
        assert get_best_action_from_q(q_0, s1, actions0) == a0
        assert get_best_action_from_q(q_0, s2, actions0) == a2
        assert get_best_action_from_q(q_0, s3, actions0) in actions0
        assert get_best_action_from_q(q_1, s0, actions0) == a0
        assert get_best_action_from_q(q_2, s0, actions0) == a2
        assert get_best_action_from_q(q_2, s0, [a0, a1]) == a1

    @staticmethod
    def test_get_random_policy():
        p0 = get_random_policy(actions0)
        assert np.isclose(p0(s0, a0), 1/3)
        assert np.isclose(p0(s0, a1), 1 / 3)
        assert np.isclose(p0(s0, a2), 1 / 3)
        assert np.isclose(p0(s0, a3), 0)
        assert np.isclose(p0(s2, a2), 1 / 3)
