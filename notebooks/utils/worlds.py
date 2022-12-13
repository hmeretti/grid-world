from grid_world.grid_world import GridWorld


# these are some worlds we will use often
small_world_01 = GridWorld(
    grid_shape=(4, 5),
    terminal_states_coordinates=((0, 4),),
    walls_coordinates=((0, 1), (1, 1), (2, 3)),
    traps_coordinates=((1, 3),),
)

small_world_02 = GridWorld(
    grid_shape=(5, 5),
    terminal_states_coordinates=((1, 4),),
    walls_coordinates=((0, 1), (1, 1), (2, 3), (3, 3)),
    traps_coordinates=((1, 3),),
)

medium_world_01 = GridWorld(
    grid_shape=(6, 7),
    terminal_states_coordinates=((5, 6),),
    walls_coordinates=(
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (4, 3),
        (5, 5),
        (3, 3),
        (3, 4),
        (3, 5),
    ),
    traps_coordinates=((1, 3), (2, 3), (0, 5), (1, 5)),
)


large_world_01 = GridWorld(
    grid_shape=(8, 10),
    terminal_states_coordinates=((5, 6),),
    walls_coordinates=(
        (1, 1),
        (2, 1),
        (3, 1),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 5),
        (4, 5),
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
        (2, 6),
        (1, 6),
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (6, 6),
        (6, 7),
        (6, 8),
        (5, 8),
        (3, 8),
        (3, 9),
        (0, 3),
    ),
    traps_coordinates=((1, 3), (1, 4), (1, 9)),
)
