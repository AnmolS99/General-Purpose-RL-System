import numpy as np
import random
import tensorflow as tf


class Critic():

    def __init__(self, use_nn, nn, lr, elig_decay, disc_factor) -> None:
        self.use_nn = use_nn

        if self.use_nn:
            self.nn = self.create_nn(nn)
        else:
            self.nn = None
        self.lr = lr  # Learning rate
        self.elig_decay = elig_decay  # Eligibility decay
        self.disc_factor = disc_factor  # Discount factor

        # Initializing V(s) as empty dictionary
        self.state_values = {}

        # Initializing e(s) as empty dictionary
        self.elig = {}

    def create_nn(self, nn):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="tanh"),
            tf.keras.layers.Dense(16, activation="tanh"),
            tf.keras.layers.Dense(1)
        ])
        return model

    def get_state_value(self, s):
        """
        Returns the value of a state
        """
        # If no value is found for the state, return a small random number
        # IMPORTANT: s is assumed to be a list and is therefore converted to tuple
        return self.state_values.get(tuple(s), random.random() * 0.5)

    def get_elig_value(self, s):
        """
        Returns the eligibility trace value for a state
        """
        # If no value is found for the elibility trace of the state, return 0
        # IMPORTANT: s is assumed to be a list and is therefore converted to tuple
        return self.elig.get(tuple(s), 0)

    def reset_elig(self):
        """
        Resetting eligibilities by setting it to an empty dictionary
        """
        self.elig = {}

    def calculate_td_error(self, r, s, s_next):
        return r + self.disc_factor * self.get_state_value(
            s_next) - self.get_state_value(s)


if __name__ == "__main__":
    critic = Critic(True, 1, 0.5, 0.99)
    states = np.random.uniform(size=(10, 9)) < 0.3
    a = critic.nn(states[0])
