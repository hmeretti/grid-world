import numpy as np
import pytest

from exploring_agents.commons.dacaying_functions import ExpDecay
from policies import EpsilonGreedy
from tests.constants.actions import a0, a1, a2, a3
from tests.constants.states import s0, s1, s2


class TestEpsilonGreedy:
    @staticmethod
    def test_call_update_and_decay():
        epsilon = 0.1
        decay_func = ExpDecay()
        test_policy = EpsilonGreedy(
            epsilon=epsilon, actions=(a0, a1, a2), epsilon_decay=decay_func
        )

        # test on empty states
        assert np.isclose(test_policy(s0, a0), 1 / 3)
        assert np.isclose(test_policy(s1, a0), 1 / 3)
        assert np.isclose(test_policy(s0, a1), 1 / 3)
        assert np.isclose(test_policy(s0, a2), 1 / 3)
        with pytest.raises(ValueError):
            test_policy(s0, a3)

        # update a state
        test_policy.update(s0, a1)
        assert np.isclose(test_policy(s0, a0), epsilon / 2)
        assert np.isclose(test_policy(s1, a0), 1 / 3)
        assert np.isclose(test_policy(s0, a1), 1 - epsilon)
        assert np.isclose(test_policy(s0, a2), epsilon / 2)

        # make new update on same state
        test_policy.update(s0, a2)
        assert np.isclose(test_policy(s0, a0), epsilon / 2)
        assert np.isclose(test_policy(s0, a1), epsilon / 2)
        assert np.isclose(test_policy(s0, a2), 1 - epsilon)

        # decay once
        test_policy.decay()
        decayed_epsilon = decay_func(epsilon)
        assert np.isclose(test_policy(s0, a0), decayed_epsilon / 2)
        assert np.isclose(test_policy(s0, a1), decayed_epsilon / 2)
        assert np.isclose(test_policy(s0, a2), 1 - decayed_epsilon)
        assert np.isclose(test_policy(s1, a0), 1 / 3)

        # update other state
        test_policy.update(s1, a0)
        assert np.isclose(test_policy(s1, a0), 1 - decayed_epsilon)
        assert np.isclose(test_policy(s1, a1), decayed_epsilon / 2)
        assert np.isclose(test_policy(s1, a2), decayed_epsilon / 2)

        # decay again
        test_policy.decay()
        decayed_epsilon = decay_func(decayed_epsilon)
        assert np.isclose(test_policy(s0, a0), decayed_epsilon / 2)
        assert np.isclose(test_policy(s0, a1), decayed_epsilon / 2)
        assert np.isclose(test_policy(s0, a2), 1 - decayed_epsilon)
        assert np.isclose(test_policy(s1, a0), 1 - decayed_epsilon)
        assert np.isclose(test_policy(s2, a0), 1 / 3)

        # reset epsilon
        new_epsilon = 0.2
        test_policy.epsilon = new_epsilon
        assert np.isclose(test_policy(s0, a0), new_epsilon / 2)
        assert np.isclose(test_policy(s0, a1), new_epsilon / 2)
        assert np.isclose(test_policy(s0, a2), 1 - new_epsilon)
        assert np.isclose(test_policy(s1, a0), 1 - new_epsilon)
        assert np.isclose(test_policy(s2, a0), 1 / 3)
