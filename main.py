import sys
from matplotlib import pyplot as plt
from gambler import GamblerSimWorld
from pole_balancing import PoleBalancingSimWorld
from rl_system import RLSystem
from towers_of_hanoi import TowersOfHanoiSimWorld


def show_optimal_state_action_gambler(gsw, rls):
    wager = []
    for s in range(1, 101):
        one_hot_s = gsw.one_hot_encode(s)
        wager.append(
            rls.actor.get_optimal_action(one_hot_s,
                                         gsw.get_valid_actions(one_hot_s)))
    plt.plot(wager)
    plt.xlabel("State")
    plt.ylabel("Wager")
    plt.vlines(x=[12.5, 25, 37.5, 50, 62.5, 75, 87.5],
               ymin=[0, 0, 0, 0, 0, 0, 0],
               ymax=[12.5, 25, 37.5, 50, 37.5, 25, 12.5],
               colors="red",
               linestyles="dotted")
    plt.show()


def main(sim_world, is_critic_ann):

    if (sim_world == "pbsw"):
        pbsw = PoleBalancingSimWorld(l=0.5, m_p=0.1, g=-9.8, timestep=0.02)
        if (not is_critic_ann):
            # Pole balancing with table-based critic
            rls = RLSystem(pbsw, num_episodes=200, max_steps=300, critic_use_nn=False, critic_nn_specs=None, actor_lr=0.3, critic_lr=0.3,actor_elig_decay= 0.5,critic_elig_decay=0.5,actor_disc_factor=0.99,critic_disc_factor=0.99,
                epsilon=0.5,epsilon_decay_rate=0.04,display=True,delay=1)
        else:
            # Pole balancing with ANN-based critic
            rls = RLSystem(pbsw, 50, 300, True, (20, 20, 20, 40, 5, 1), 0.3, 0.3, 0.5,
                           0.5, 0.5, 0.5, 0.5, 0.1, True, 1)
    elif (sim_world == "tohsw"):
        num_pegs = 3
        num_discs = 4
        tohsw = TowersOfHanoiSimWorld(num_pegs, num_discs)
        if (not is_critic_ann):
            # Towers of Hanoi with table-based critic
            rls = RLSystem(tohsw, 100, 300, False, None, 0.3, 0.03, 0.5, 0.5, 0.99,
                           0.99, 0.9, 0.03, True, 1)
        else:
            # Towers of Hanoi with ANN-based critic
            rls = RLSystem(tohsw, 10, 300, True, (num_pegs * num_discs, 30, 30, 30, 1),
                           2, 0.1, 0.5, 0.5, 0.99, 0.99, 0.9, 0.03, True, 1)
    elif (sim_world == "gsw"):
        gsw = GamblerSimWorld(0.6)
        if (not is_critic_ann):
            # The Gambler with table-based critic
            rls = RLSystem(gsw, 100, 300, False, None, 0.5, 0.05, 0.5, 0.5, 1, 1, 0.9,
                        0.075, True, 1)
        else:
            # The Gambler with ANN-based critic
            rls = RLSystem(gsw, 10, 300, True, (101, 5, 1), 0.08, 0.01, 0.5, 0.5, 1, 1,
                           0.9, 0.075, True, 1)
    else:
        raise Exception()

    rls.generic_actor_critic_algorithm()

    if sim_world == "gsw":
        show_optimal_state_action_gambler(gsw, rls)


if __name__ == "__main__":
    # sim_world = str(sys.argv[1]).lower()
    # if (len(sys.argv)) >= 3:
    #     is_critic_ann = (str(sys.argv[2]).lower() == "ann")
    # else:
    #     is_critic_ann = False


    main(sim_world="tohsw", is_critic_ann=False)
