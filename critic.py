import numpy as np
import random


class Critic():

    def __init__(self, use_nn, nn, lr, elig_decay, disc_factor) -> None:
        self.use_nn = use_nn
        self.nn = nn
        self.lr = lr  # Learning rate
        self.elig_decay = elig_decay  # Eligibility decay
        self.disc_factor = disc_factor  # Discount factor

        # Initializing V(s) as empty dictionary
        self.state_values = {}

        # Initializing e(s) as empty dictionary
        self.elig = {}

    def get_state_value(self, s):
        """
        Returns the value of a state
        """
        # If no value is found for the state, return a small random number
        return self.state_values.get(s, random.random() * 0.5)

    def get_elig_value(self, s):
        """
        Returns the eligibility trace value for a state
        """
        # If no value is found for the elibility trace of the state, return 0
        return self.elig.get(s, 0)

    def reset_elig(self):
        """
        Resetting eligibilities by setting it to an empty dictionary
        """
        self.elig = {}

    def calculate_td_error(self, r, s, s_next):
        return r + self.disc_factor * self.get_state_value(
            s_next) - self.get_state_value(s)
