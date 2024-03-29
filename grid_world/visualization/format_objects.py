import numpy as np

from grid_world.grid_world import GridWorld
from abstractions import PolicyRec, StateEvalDict
from grid_world.visualization.unicode_definitions import states_symbols


def get_policy_rec_str(d: PolicyRec, world: GridWorld) -> str:
    dict_str = ""
    for j in reversed(range(world.grid_shape[1])):
        for i in range(world.grid_shape[0]):
            if (i, j) in world.walls_coordinates:
                cur_char = states_symbols["wall"]
            else:
                cur_state = world.get_state((i, j))
                if cur_state.kind in ["trap", "terminal"]:
                    cur_char = cur_state.get_unicode()
                else:
                    cur_char = d[cur_state].unicode
            dict_str += f"{cur_char}"
        dict_str += "\n\n"

    return dict_str


def get_policy_eval_str(v: StateEvalDict, world: GridWorld) -> str:
    v0 = {x: f"{v[x]:.2f}" for x in v}
    ml = np.max([len(x) for x in v0.values()])
    spaces = " " * ml
    v0 = {x: (ml - len(v0[x])) * " " + v0[x] for x in v0}
    dict_str = ""
    for j in reversed(range(world.grid_shape[1])):
        for i in range(world.grid_shape[0]):
            if (i, j) in world.walls_coordinates:
                cur_char = f" {spaces} "
            elif (s := world.get_state((i, j))) in v.keys():
                cur_char = f" {v0[s]} "
            else:
                cur_char = f" {spaces} "
            dict_str += f" {cur_char} "
        dict_str += "\n\n"

    return dict_str


def get_world_str(
    world: GridWorld,
    agent_position: tuple[int, int] = None,
    show_coordinates: bool = True,
) -> str:
    """
    creates a string visualization of an world and agent

    :param world: the world to be represented
    :param agent_position: position of agent in the world
    :param show_coordinates: show coordinates under axes, may break for large worlds
    :return: string visualization of the world
    """
    world_str = ""
    for j in reversed(range(world.grid_shape[1])):
        if show_coordinates:
            world_str += f"{j}"
        for i in range(world.grid_shape[0]):
            if (i, j) == agent_position:
                cur_char = states_symbols["agent"]
            elif (i, j) in world.walls_coordinates:
                cur_char = states_symbols["wall"]
            else:
                cur_state = world.get_state((i, j))
                cur_char = cur_state.get_unicode()

            world_str += f"{cur_char}"
        world_str += f"\n\n"
    if show_coordinates:
        world_str += " " + "".join([f" {i} " for i in range(world.grid_shape[0])])

    return world_str


def get_world_str_lines(
    world: GridWorld,
    agent_position: tuple[int, int] = None,
    show_coordinates: bool = False,
) -> list[str]:
    """
    creates a line by line string visualization of an world and agent. Useful for animations with curses

    :param world: the world to be represented
    :param agent_position: position of agent in the world
    :param show_coordinates: show coordinates under axes, may break for large worlds
    :return: string visualization of the world
    """
    lines = []
    for j in reversed(range(world.grid_shape[1])):
        line_str = ""
        if show_coordinates:
            line_str += f"{j}"
        for i in range(world.grid_shape[0]):
            if (i, j) == agent_position:
                cur_char = states_symbols["agent"]
            elif (i, j) in world.walls_coordinates:
                cur_char = states_symbols["wall"]
            else:
                cur_state = world.get_state((i, j))
                cur_char = cur_state.get_unicode()

            line_str += f"{cur_char}"
        lines.append(line_str)
    if show_coordinates:
        lines.append(" " + "".join([f" {i} " for i in range(world.grid_shape[0])]))

    return lines
