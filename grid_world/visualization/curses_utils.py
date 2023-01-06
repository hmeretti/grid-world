import time
import cursor
import curses

from grid_world.action import GWorldAction
from grid_world.grid_world import GridWorld
from grid_world.state import GWorldState
from grid_world.visualization.format_objects import get_world_str_lines
from grid_world.visualization.unicode_definitions import states_symbols


def animate_episodes(
    world: GridWorld,
    states_history: list[list[GWorldState]],
    actions_history: list[list[GWorldAction]],
    sleep_time: float = 1,
    episodes_to_animate: list[int] = None,
):
    # start visual stuff
    console = curses.initscr()
    cursor.hide()

    for episode in (
        episodes_to_animate
        if episodes_to_animate is not None
        else range(len(states_history))
    ):
        console.clrtobot()
        draw_world(world, console)
        states = states_history[episode]
        for step, s in enumerate(states):
            action_char = actions_history[episode][step - 1].unicode if step > 0 else ""
            draw_point(
                *coordinates_to_drawing_position(*s.coordinates, world.grid_shape),
                console,
            )
            draw_line(f"episode: {episode}", world.grid_shape[1], console)
            draw_line(f"step: {step}", world.grid_shape[1] + 1, console)
            draw_line(
                f"action: {action_char}",
                world.grid_shape[1] + 2,
                console,
            )
            console.refresh()
            time.sleep(sleep_time)
            restore_world_coordinate(*s.coordinates, world, console)

    time.sleep(3 * sleep_time)
    cursor.show()
    curses.endwin()


def draw_world(world, console):
    console.clear()
    lines = get_world_str_lines(world)
    for i, line in enumerate(lines):
        console.addstr(i, 0, line)
    console.refresh()


def draw_line(text, line_idx, console):
    console.addstr(line_idx, 0, text)


def draw_point(x, y, console, char=states_symbols["agent"]):
    console.addstr(x, y, char)


def coordinates_to_drawing_position(x, y, grid_shape):
    return grid_shape[1] - y - 1, 3 * x


def restore_world_coordinate(x, y, world, console):
    draw_point(
        *coordinates_to_drawing_position(x, y, world.grid_shape),
        console,
        world.get_state((x, y)).get_unicode(),
    )


def animate_tag_episode(
    world: GridWorld,
    states_1: list[GWorldState],
    actions_1: list[GWorldAction],
    states_2: list[GWorldState],
    actions_2: list[GWorldAction],
    sleep_time: float = 1,
):
    # start visual stuff
    console = curses.initscr()
    cursor.hide()

    console.clrtobot()
    draw_world(world, console)
    for step, s1 in enumerate(states_1):
        draw_line(f"step: {step}", world.grid_shape[1], console)

        if step > 0:
            restore_world_coordinate(*states_1[step - 1].coordinates, world, console)
        _draw_point_info_for_agent(
            console,
            world,
            s1,
            actions_1[step - 1].unicode if step > 0 else "",
            states_symbols["agent"],
            world.grid_shape[1] + 1,
            "Agent 1 action",
        )
        console.refresh()

        # we'll add this, so we don't print a weird move that is saved
        if step > 0 and s1.coordinates == s2.coordinates:
            break

        s2 = states_2[step]
        if step > 0:
            time.sleep(sleep_time)
            restore_world_coordinate(*states_2[step - 1].coordinates, world, console)

        _draw_point_info_for_agent(
            console,
            world,
            s2,
            actions_2[step - 1].unicode if step > 0 else "",
            states_symbols["agent2"],
            world.grid_shape[1] + 2,
            "Agent 2 action",
        )
        console.refresh()
        time.sleep(sleep_time)

    draw_line(
        f"Agent 1 wins" if s1.coordinates == s2.coordinates else "Agent 2 wins",
        world.grid_shape[1] + 3,
        console,
    )
    console.refresh()
    time.sleep(10 * sleep_time)
    cursor.show()
    curses.endwin()


def _draw_point_info_for_agent(
    console, world, state, action_str, char, info_pos, label
):
    draw_point(
        *coordinates_to_drawing_position(*state.coordinates, world.grid_shape),
        console,
        char,
    )
    draw_line(
        f"{label}: {action_str}",
        info_pos,
        console,
    )
