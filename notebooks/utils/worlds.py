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

small_world_03 = GridWorld(
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
        (0, 3),
        (1, 1),
        (1, 6),
        (2, 1),
        (2, 6),
        (3, 1),
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 8),
        (3, 9),
        (4, 5),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 5),
        (5, 8),
        (6, 6),
        (6, 7),
        (6, 8),
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
    ),
    traps_coordinates=((1, 3), (1, 4), (1, 9)),
)

large_world_02 = GridWorld(
    grid_shape=(19, 19),
    terminal_states_coordinates=((7, 12),),
    walls_coordinates=(
        (0, 1),
        (1, 8),
        (1, 9),
        (1, 10),
        (1, 11),
        (1, 12),
        (1, 14),
        (1, 15),
        (1, 16),
        (1, 17),
        (2, 1),
        (2, 3),
        (2, 4),
        (2, 6),
        (2, 7),
        (2, 8),
        (2, 10),
        (2, 17),
        (3, 1),
        (3, 10),
        (3, 17),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
        (4, 5),
        (4, 10),
        (4, 17),
        (5, 5),
        (5, 10),
        (5, 17),
        (6, 5),
        (6, 7),
        (6, 8),
        (6, 9),
        (6, 10),
        (6, 11),
        (6, 12),
        (6, 17),
        (7, 5),
        (7, 7),
        (7, 17),
        (8, 5),
        (8, 6),
        (8, 7),
        (8, 12),
        (8, 14),
        (8, 15),
        (8, 17),
        (9, 5),
        (9, 12),
        (9, 17),
        (10, 5),
        (10, 12),
        (10, 13),
        (10, 14),
        (10, 16),
        (10, 17),
        (11, 5),
        (11, 14),
        (12, 5),
        (12, 14),
        (12, 17),
        (12, 18),
        (13, 5),
        (13, 14),
        (14, 5),
        (14, 14),
        (15, 5),
        (16, 5),
    ),
    traps_coordinates=(
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (2, 14),
        (2, 16),
        (3, 12),
        (3, 13),
        (4, 7),
        (4, 8),
        (4, 12),
        (4, 13),
        (4, 15),
        (5, 15),
        (6, 13),
        (6, 14),
        (6, 15),
    ),
)

# lets have some worlds for tagging
tagging_world_00 = GridWorld(
    grid_shape=(6, 6),
    initial_state_coordinates=(0, 0),
    initial_state_coordinates_2=(5, 5),
)


tagging_world_01 = GridWorld(
    grid_shape=(6, 6),
    walls_coordinates=((0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (3, 3)),
    initial_state_coordinates=(0, 0),
    initial_state_coordinates_2=(5, 5),
)

tagging_world_02 = GridWorld(
    grid_shape=(8, 10),
    initial_state_coordinates=(0, 0),
    initial_state_coordinates_2=(5, 6),
    walls_coordinates=(
        (0, 3),
        (1, 1),
        (1, 3),
        (1, 4),
        (1, 6),
        (1, 9),
        (2, 1),
        (2, 6),
        (3, 1),
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 8),
        (3, 9),
        (4, 5),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 5),
        (5, 8),
        (6, 5),
        (6, 6),
        (6, 7),
        (6, 8),
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
    ),
)
