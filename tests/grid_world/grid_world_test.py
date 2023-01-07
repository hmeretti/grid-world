from grid_world.action import GWorldAction
from tests.constants.grid_worlds import test_world_01


class TestGridWorld:
    @staticmethod
    def test_get_state():
        s00 = test_world_01.get_state((0, 0))
        s13 = test_world_01.get_state((1, 3))
        s04 = test_world_01.get_state((0, 4))
        s10 = test_world_01.get_state((1, 0))
        assert s00.coordinates == (0, 0) and s00.kind == "initial"
        assert s13.coordinates == (1, 3) and s13.kind == "trap"
        assert s04.coordinates == (0, 4) and s04.kind == "terminal"
        assert s10.coordinates == (1, 0) and s10.kind == "empty"

        # walls are not valid states
        try:
            _ = test_world_01.get_state((1, 1))
        except KeyError:
            pass

    @staticmethod
    def test_take_action():
        s00 = test_world_01.get_state((0, 0))
        s10 = test_world_01.get_state((1, 0))
        s12 = test_world_01.get_state((1, 2))
        s22 = test_world_01.get_state((2, 2))
        s32 = test_world_01.get_state((3, 2))
        s21 = test_world_01.get_state((2, 1))
        s33 = test_world_01.get_state((3, 3))
        s31 = test_world_01.get_state((3, 1))
        s03 = test_world_01.get_state((0, 3))
        s04 = test_world_01.get_state((0, 4))
        s13 = test_world_01.get_state((1, 3))
        s02 = test_world_01.get_state((0, 2))

        # basic action at 00
        assert test_world_01.take_action(s00, GWorldAction.left) == (s00, 0)
        assert test_world_01.take_action(s00, GWorldAction.up) == (s00, 0)
        assert test_world_01.take_action(s00, GWorldAction.right) == (s10, 0)
        assert test_world_01.take_action(s00, GWorldAction.down) == (s00, 0)

        # basic action at 22
        assert test_world_01.take_action(s22, GWorldAction.left) == (s12, 0)
        assert test_world_01.take_action(s22, GWorldAction.up) == (s22, 0)
        assert test_world_01.take_action(s22, GWorldAction.right) == (s32, 0)
        assert test_world_01.take_action(s22, GWorldAction.down) == (s21, 0)

        # diagonals and wait actions at 22
        assert test_world_01.take_action(s22, GWorldAction.up_right) == (s33, 0)
        assert test_world_01.take_action(s22, GWorldAction.up_left) == (s13, -1)
        assert test_world_01.take_action(s22, GWorldAction.down_right) == (s31, 0)
        assert test_world_01.take_action(s22, GWorldAction.down_left) == (s22, 0)
        assert test_world_01.take_action(s22, GWorldAction.wait) == (s22, 0)

        # basic action at 03
        assert test_world_01.take_action(s03, GWorldAction.left) == (s03, 0)
        assert test_world_01.take_action(s03, GWorldAction.up) == (s04, 1)
        assert test_world_01.take_action(s03, GWorldAction.right) == (s13, -1)
        assert test_world_01.take_action(s03, GWorldAction.down) == (s02, 0)

        # any action inside trap
        assert test_world_01.take_action(s13, GWorldAction.left) == (s00, 0)
        assert test_world_01.take_action(s13, GWorldAction.up) == (s00, 0)
        assert test_world_01.take_action(s13, GWorldAction.right) == (s00, 0)
        assert test_world_01.take_action(s13, GWorldAction.down) == (s00, 0)
        assert test_world_01.take_action(s13, GWorldAction.up_right) == (s00, 0)
        assert test_world_01.take_action(s13, GWorldAction.up_left) == (s00, 0)
        assert test_world_01.take_action(s13, GWorldAction.down_right) == (s00, 0)
        assert test_world_01.take_action(s13, GWorldAction.down_left) == (s00, 0)
        assert test_world_01.take_action(s13, GWorldAction.wait) == (s00, 0)
