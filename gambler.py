import random
import re
import numpy as np


class GamblerSimWorld:

    def __init__(self, p_w) -> None:
        self.p_w = p_w  # The win probability
        self.units = None  # Units the gambler has
        self.steps_taken = None  # The amount of gambles done

    def begin_episode(self):
        # Randomly choosing a number of monetary units between 1 and 99
        self.units = np.random.randint(1, 100)

        # Resetting the amounts of steps taken
        self.steps_taken = 0

        return self.units

    def next_state(self, action):
        """
        Goes to the next state given the action, in other words gambling "action" amount of units, and either
        winning that amount or losing it
        """
        reward = 0
        # Performing the gamble
        if random.random() <= self.p_w:
            self.units += action
            #reward = action
        else:
            self.units -= action
            #reward = -action

        if self.units == 100:
            reward = 1

        # Incrementing the amount of steps taken
        self.steps_taken += 1

        return self.units, reward

    def get_valid_actions(self, state):
        """
        Getting a list of units the gamble can gamble given the current state (units it has)
        """
        if state == 100 or state == 0:
            return [0]
        return [i for i in range(1, (min(state, 100 - state) + 1))]

    def is_end_state(self):
        """
        Checks if current state is an end state.
        The gambler has reached an end state if it either:
            1. Has reached 100 units, or
            2. Lost all of its units
        """
        if self.units == 100 or self.units == 0:
            return True
        return False

    def end_episode(self):
        return

    def show_best_history(self):
        return


if __name__ == "__main__":
    gsw = GamblerSimWorld(0.5)
    gsw.begin_episode()
    gsw.next_state(10)
    print(gsw.get_valid_actions(3))
