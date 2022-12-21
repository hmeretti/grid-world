from grid_world.action import Action


# this is the basic reward function we will use most of the time
def basic_reward(e):
    if e == 1:
        return 0
    elif e == -1:
        return -100
    else:
        return -1


# most of the time we will work with this limited set of actions
basic_actions = [Action.up, Action.down, Action.left, Action.right]
