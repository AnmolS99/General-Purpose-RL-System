from matplotlib import pyplot as plt
from gambler import GamblerSimWorld
from pole_balancing import PoleBalancingSimWorld
from rl_system import RLSystem
from towers_of_hanoi import TowersOfHanoiSimWorld
import random
import numpy as np
import tensorflow as tf
from time import time


def show_optimal_state_action_gambler(gsw, rls):
    wager = []
    for s in range(1, 101):
        one_hot_s = gsw.one_hot_encode(s)
        wager.append(
            rls.actor.get_optimal_action(one_hot_s,
                                         gsw.get_valid_actions(one_hot_s)))
    plt.plot(wager)
    plt.vlines(x=[12.5, 25, 37.5, 50, 62.5, 75, 87.5],
               ymin=[0, 0, 0, 0, 0, 0, 0],
               ymax=[12.5, 25, 37.5, 50, 37.5, 25, 12.5],
               colors="red",
               linestyles="dotted")
    plt.show()


if __name__ == "__main__":

    # Setting random seeds to be able to reproduce results
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    pbsw = PoleBalancingSimWorld()
    num_pegs = 3
    num_discs = 3
    tohsw = TowersOfHanoiSimWorld(num_pegs, num_discs)
    gsw = GamblerSimWorld(0.4)

    # Pole balancing with table-based critic
    # rls = RLSystem(pbsw, 200, 300, False, None, 0.3, 0.3, 0.5, 0.5, 0.99, 0.99,
    #                0.5, 0.04, False, 1)

    # Pole balancing with ANN-based critic
    # rls = RLSystem(pbsw, 100, 300, True, (20, 16, 16, 1), 0.3, 0.3, 0.5, 0.5,
    #                0.99, 0.99, 0.5, 0.1, False, 1)

    # Towers of Hanoi with table-based critic
    # rls = RLSystem(tohsw, 300, 300, False, None, 0.3, 0.03, 0.5, 0.5, 0.99,
    #                0.99, 0.9, 0.05, False, 1)

    # Towers of Hanoi with ANN-based critic
    # rls = RLSystem(tohsw, 300, 300, True, (num_pegs * num_discs, 16, 16, 1),
    #                0.2, 0.01, 0.5, 0.5, 0.99, 0.99, 0.9, 0.03, False, 1)

    # The Gambler with table-based critic
    # rls = RLSystem(gsw, 25000, 300, False, None, 0.5, 0.05, 0.5, 0.5, 1, 1,
    #                0.9, 0.075, False, 1)

    # The Gambler with ANN-based critic
    # rls = RLSystem(gsw, 1000, 300, True, (101, 16, 16, 1), 0.1, 0.05, 0.5, 0.5,
    #                1, 1, 0.9, 0.075, False, 1)
    rls.generic_actor_critic_algorithm()

    if rls.sim_world == gsw:
        show_optimal_state_action_gambler(gsw, rls)