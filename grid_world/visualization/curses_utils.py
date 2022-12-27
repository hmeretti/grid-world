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
            draw_point(
                *coordinates_to_drawing_position(*s.coordinates, world.grid_shape),
                console,
            )
            draw_line(f"episode: {episode}", world.grid_shape[1], console)
            draw_line(f"step: {step}", world.grid_shape[1] + 1, console)
            draw_line(
                f"action: {actions_history[episode][step].unicode}",
                world.grid_shape[1] + 2,
                console,
            )
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
    console.refresh()


def coordinates_to_drawing_position(x, y, grid_shape):
    return grid_shape[1] - y - 1, 3 * x


def restore_world_coordinate(x, y, world, console):
    draw_point(
        *coordinates_to_drawing_position(x, y, world.grid_shape),
        console,
        world.get_state((x, y)).get_unicode(),
    )
