from grid_world.action import GWorldAction


# this is the basic reward function we will use most of the time
def basic_reward(e):
    if e == 1:
        return 0
    elif e == -1:
        return -100
    else:
        return -1


# this is the basic reward function for an agent trying to prolong an episode
def basic_tag_reward(e):
    if e == 1:
        return 1
    elif e == -1:
        return -1
    else:
        return 0


# most of the time we will work with this limited set of actions
basic_actions = (
    GWorldAction.up,
    GWorldAction.down,
    GWorldAction.left,
    GWorldAction.right,
)
