import numpy as np

from exploring_agents.policies.epsilon_greedy import EpsilonGreedy
from tests.constants.actions import a0, a1, a2, a3
from tests.constants.states import s0, s1, s2, s3
from tests.constants.worlds import RandomWorld
from utils.policy import get_best_action_from_q, get_random_policy, get_policy_rec

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
        assert np.isclose(p0(s0, a0), 1 / 3)
        assert np.isclose(p0(s0, a1), 1 / 3)
        assert np.isclose(p0(s0, a2), 1 / 3)
        assert np.isclose(p0(s0, a3), 0)
        assert np.isclose(p0(s2, a2), 1 / 3)

    @staticmethod
    def test_get_policy_rec():
        epsilon = 0.1
        test_policy = EpsilonGreedy(epsilon=epsilon, actions=actions0)
        test_world = RandomWorld(states0, s0)

        # set some action to be the best
        test_policy.update(s0, a1)
        test_policy.update(s1, a2)
        police_rec_0 = get_policy_rec(test_policy, test_world, actions0)
        assert police_rec_0[s0] == a1
        assert police_rec_0[s1] == a2
        assert police_rec_0[s2] in actions0

        # change one of the action
        test_policy.update(s0, a0)
        police_rec_0 = get_policy_rec(test_policy, test_world, actions0)
        assert police_rec_0[s0] == a0
        assert police_rec_0[s1] == a2
        assert police_rec_0[s2] in actions0
