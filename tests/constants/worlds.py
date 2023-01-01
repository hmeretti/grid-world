import random

from abstractions import World


class RandomWorld(World):
    def __init__(
        self,
        states,
        initial_state,
    ):
        self.states = states
        self.initial_state = initial_state

    def take_action(self, state, action):
        return random.choice(self.states)
