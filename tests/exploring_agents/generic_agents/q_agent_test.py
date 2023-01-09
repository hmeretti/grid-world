import numpy as np

from exploring_agents import QAgent
from notebooks.utils.basics import basic_reward
from tests.constants.actions import a0, a1, a2, a3
from tests.constants.states import s0, s1


class TestQAgent:
    @staticmethod
    def test_select_and_update():
        # This is a stochastic test, which is subject to random failure, although very unlikely

        gamma = 0.9
        alpha = 0.1
        epsilon = 0.1
        iterations = 1e5

        agent1 = QAgent(
            reward_function=basic_reward,
            actions=(a0, a1, a2),
            gamma=gamma,
            alpha=alpha,
            epsilon=epsilon,
            q_0={(s1, a0): 1, (s1, a1): 0.1, (s1, a2): -1},
        )

        # agent selects actions randomly at state with no updates
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

        # agent selects best action with probability (1-epsilon) and
        # the other with probability epsilon/(len(actions) -1)
        actions = [agent1.select_action(s1) for _ in range(int(iterations))]
        assert np.isclose(
            np.sum([x == a0 for x in actions]) / iterations, 1 - epsilon, atol=1e-2
        )
        assert np.isclose(
            np.sum([x == a1 for x in actions]) / iterations, epsilon / 2, atol=1e-2
        )
        assert np.isclose(
            np.sum([x == a2 for x in actions]) / iterations, epsilon / 2, atol=1e-2
        )

        # let's simulate an update where agent takes action a1 in state s0 with effect 1 and goes to state s1
        reward = agent1.run_update(s0, a1, 1, s1)
        assert reward == basic_reward(1)

        # this should make a1 the best action at s0 so
        actions = [agent1.select_action(s0) for _ in range(int(iterations))]
        assert np.isclose(
            np.sum([x == a0 for x in actions]) / iterations, epsilon / 2, atol=1e-2
        )
        assert np.isclose(
            np.sum([x == a1 for x in actions]) / iterations, 1 - epsilon, atol=1e-2
        )
        assert np.isclose(
            np.sum([x == a2 for x in actions]) / iterations, epsilon / 2, atol=1e-2
        )

        # more precisely, a0 is the best action for s1 so
        assert agent1.q[(s0, a1)] == alpha * (
            basic_reward(1) + gamma * agent1.q[(s1, a0)]
        )
