import numpy as np
import matplotlib.pyplot as plt


class TowersOfHanoiSimWorld:
    """
    Tower of Hanoi simulation world
    """

    def __init__(self, num_pegs=3, num_discs=3) -> None:
        self.num_pegs = num_pegs
        self.num_discs = num_discs
        # List of what peg (represented by the value in the list) each disc (given by the index) is on
        # The lower indices have the larger discs
        self.discs_on_pegs = []
        self.steps_taken = 0
        self.history = []
        self.best_episode_history = []

    def begin_episode(self):
        """
        Beginning a new episode
        """
        # Setting all the discs on peg 0
        self.discs_on_pegs = [0] * self.num_discs

        # Resetting history and adding current state
        self.history = []
        self.history.append(self.discs_on_pegs.copy())

        # Restting steps taken
        self.steps_taken = 0

        return self.one_hot_encode(tuple(self.discs_on_pegs))

    def next_state(self, action):
        """
        Goes to the next state given the action, which is on the form (disc_nr, peg_nr)
        """
        disc_nr = action[0]
        peg_nr = action[1]
        self.discs_on_pegs[disc_nr] = peg_nr

        # Setting reward
        if self.is_end_state():
            reward = 1000000
        else:
            reward = -1

        # Adding current state to history of this episode
        self.history.append(self.discs_on_pegs.copy())

        # Incrementing the steps taken
        self.steps_taken += 1

        return self.one_hot_encode(tuple(self.discs_on_pegs)), reward

    def get_valid_actions(self, state):
        """
        Getting a list of valid actions in a given state
        """
        state = self.rev_one_hot_encode(state)
        valid_actions = []
        # Checking the valid actions for each disc
        for disc in range(len(state)):
            curr_peg = state[disc]

            # If there are any discs (smaller than current discs) on the same peg, then there are no possible actions
            if curr_peg in state[disc + 1:]:
                continue

            # Iterating through all pegs except the one the current disc is on
            pegs_list = [i for i in range(self.num_pegs)]
            pegs_list.remove(curr_peg)

            for peg in pegs_list:

                # If there is a smaller disc (than the current one) on this peg, the current disc cannot be placed on it
                if peg in state[disc + 1:]:
                    continue
                else:
                    # Adding the move "disc_number to peg_number" in the valid actions list
                    # with the format (disc_nr, peg_nr)
                    valid_actions.append((disc, peg))

        return valid_actions

    def is_end_state(self):
        """
        Checks whether the input state is an end state or not
        """
        if len(set(self.discs_on_pegs)) == 1 and self.discs_on_pegs[0] != 0:
            return True
        return False

    def end_episode(self):
        """
        Ending the current episode by checking if the history of solving the problem is smallest (most efficient) one yet
        """
        if len(self.history) < len(
                self.best_episode_history) or not self.best_episode_history:
            self.best_episode_history = self.history

    def show_best_history(self, delay):
        """
        Showing the best history
        """
        step_nr = 0
        for step in self.best_episode_history:
            self.show_state(step, step_nr, delay)
            step_nr += 1

    def show_state(self, state, step_nr, delay):
        """
        Displaying a state 
        """
        colors = ["red", "blue", "green", "yellow", "orange", "purple"]
        figure, axes = plt.subplots()
        heights = [0] * self.num_pegs
        for i in range(self.num_discs):
            r = (self.num_discs - i) * 0.05
            if heights[state[i]] == 0:
                heights[state[i]] = r
            height = heights[state[i]]
            circle = plt.Circle(((state[i] * (1 / self.num_pegs) +
                                  (1 / self.num_pegs) / 2), height),
                                r,
                                color=colors[i % len(colors)])
            heights[state[i]] += r
            axes.add_artist(circle)
        plt.title("Towers Of Hanoi (step number: " + str(step_nr) + ")")
        plt.savefig(f"images/tohsw_step_{step_nr}.png")
        plt.show(block=False)
        plt.pause(delay)
        plt.close()

    def one_hot_encode(self, state):
        """
        Function that turns state into one hot encoding
        """
        # Converting from tuple to np.array
        state_list = np.array(list(state))

        #Defining the shape
        shape = (state_list.size, self.num_pegs)

        # Creating the one hot encoding in form of matrix
        one_hot_enc_state = np.zeros(shape)

        # Getting a list of all row indices
        rows = np.arange(state_list.size)

        # Placing a 1 in the place given by the index in state
        one_hot_enc_state[rows, state_list] = 1

        return one_hot_enc_state.flatten()

    def rev_one_hot_encode(self, state):
        """
        Reversing a one hot encoding
        """
        # Unflattening
        state_matrix = np.split(state, self.num_discs)
        rev_enc = np.argmax(state_matrix, axis=1)

        return tuple(rev_enc)
