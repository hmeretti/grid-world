import numpy as np
import pytest

from exploring_agents.policies.random_policy import RandomPolicy
from tests.constants.actions import a0, a1, a2, a3
from tests.constants.states import s0, s1


class TestRandomPolicy:
    @staticmethod
    def test_call_update_and_decay():
        test_policy = RandomPolicy(
            actions=[a0, a1, a2],
        )

        assert np.isclose(test_policy(s0, a0), 1 / 3)
        assert np.isclose(test_policy(s1, a0), 1 / 3)
        assert np.isclose(test_policy(s0, a1), 1 / 3)
        assert np.isclose(test_policy(s0, a2), 1 / 3)
        with pytest.raises(ValueError):
            test_policy(s0, a3)

        # with 4 actions now
        test_policy = RandomPolicy(
            actions=[a0, a1, a2, a3],
        )

        assert np.isclose(test_policy(s0, a0), 1 / 4)
        assert np.isclose(test_policy(s1, a0), 1 / 4)
        assert np.isclose(test_policy(s0, a1), 1 / 4)
        assert np.isclose(test_policy(s0, a2), 1 / 4)
        assert np.isclose(test_policy(s0, a3), 1 / 4)
