from actor import Actor
from critic import Critic
from gambler import GamblerSimWorld
from pole_balancing import PoleBalancingSimWorld
from matplotlib import pyplot as plt

from towers_of_hanoi import TowersOfHanoiSimWorld


class RLSystem():

    def __init__(self, sim_world, num_episodes, max_steps, critic_use_nn,
                 critic_nn, actor_lr, critic_lr, actor_elig_decay,
                 critic_elig_decay, actor_disc_factor, critic_disc_factor,
                 epsilon, epsilon_decay_rate, display, delay) -> None:
        self.sim_world = sim_world
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.display = display
        self.delay = delay

        self.critic = Critic(critic_use_nn, critic_nn, critic_lr,
                             critic_elig_decay, critic_disc_factor)
        self.actor = Actor(actor_lr, actor_elig_decay, actor_disc_factor,
                           epsilon, epsilon_decay_rate)

    def generic_actor_critic_algorithm(self):
        result_list = []

        # Repeating for each episode
        for i in range(self.num_episodes):

            # Resetting eligibilities in actor and critic
            self.actor.reset_elig()
            self.critic.reset_elig()

            # Decreasing epsilon for actor, since we want less exploration as number of episodes goes up
            if i % (self.num_episodes // 100) == 0:
                self.actor.epsilon *= (1 - self.actor.epsilon_decay_rate)
                print(self.actor.epsilon)

            # Initializing state and action
            s = self.sim_world.begin_episode()
            a = self.actor.get_action(s, self.sim_world.get_valid_actions(s))

            # Initializing a list of steps taken in the current episode on the form [(s_0, a_0), (s_1, a_1), ...]
            episode_list = []

            # Repeating for each step of the episode
            for j in range(self.max_steps):

                # Adding the state s and action a we are in to the episode_list before taking the next step
                episode_list.append((s, a))

                # Performing the action a in state s and ending up in state s_next and recieving reward r
                s_next, r = self.sim_world.next_state(a)

                # Getting the action (a_next) to do in state s_next
                a_next = self.actor.get_action(
                    s_next, self.sim_world.get_valid_actions(s_next))

                # Setting the eligibility trace in the actor to 1
                # IMPORTANT: s is assumed to be a list and is therefore converted to tuple
                self.actor.elig[(tuple(s), a)] = 1

                # Critic calculates TD-error
                td_error = self.critic.calculate_td_error(r, s, s_next)

                # Setting the eligibility trace in the critic to 1
                # IMPORTANT: s is assumed to be a list and is therefore converted to tuple
                self.critic.elig[tuple(s)] = 1

                # Going through each SAP in the current episode and updating values and policies
                for s_curr, a_curr in episode_list:

                    # Critic updates current states value
                    # IMPORTANT: s_curr is assumed to be a list and is therefore converted to tuple
                    self.critic.state_values[tuple(s_curr)] = (
                        self.critic.get_state_value(s_curr) + self.critic.lr *
                        td_error * self.critic.get_elig_value(s_curr))

                    # Critic updates eligibility trace
                    # IMPORTANT: s_curr is assumed to be a list and is therefore converted to tuple
                    self.critic.elig[tuple(s_curr)] = (
                        self.critic.disc_factor * self.critic.elig_decay *
                        self.critic.get_elig_value(s_curr))

                    # Actor updates policy
                    # IMPORTANT: s_curr is assumed to be a list and is therefore converted to tuple
                    self.actor.policy[(tuple(s_curr), a_curr)] = (
                        self.actor.get_policy(s_curr, a_curr) + self.actor.lr *
                        td_error * self.actor.get_elig_value(s_curr, a_curr))

                    # Actor updates eligibility trace
                    # IMPORTANT: s_curr is assumed to be a list and is therefore converted to tuple
                    self.actor.elig[(
                        tuple(s_curr), a_curr
                    )] = self.actor.disc_factor * self.actor.elig_decay * self.actor.get_elig_value(
                        s_curr, a_curr)

                # Setting the current state to s_next and current action to a_next
                s = s_next
                a = a_next

                # If we are in an end state, we end the episode
                if self.sim_world.is_end_state():
                    self.sim_world.end_episode()
                    break

            result_list.append(self.sim_world.steps_taken)

        # Plotting the result list
        plt.plot(result_list)
        plt.show()

        # Showing the history of the best episode
        self.sim_world.show_best_history()


if __name__ == "__main__":
    pbsw = PoleBalancingSimWorld()
    tohsw = TowersOfHanoiSimWorld(3, 4)
    gsw = GamblerSimWorld(0.5)

    # rls = RLSystem(pbsw, 200, 300, False, 1, 0.3, 0.3, 0.5, 0.5, 0.99, 0.99,
    #                0.5, 0.05, False, 1)

    rls = RLSystem(tohsw, 500, 300, False, 1, 0.3, 0.3, 0.5, 0.5, 0.99, 0.99,
                   0.5, 0.05, False, 1)

    # rls = RLSystem(gsw, 25000, 300, False, 1, 0.05, 0.05, 0.5, 0.5, 1, 1, 0.5,
    #                0.075, False, 1)
    rls.generic_actor_critic_algorithm()
    # wager = []
    # for i in range(1, 101):
    #     wager.append(rls.actor.get_optimal_action(i, gsw.get_valid_actions(i)))
    # plt.plot(wager)
    # plt.vlines(x=[12.5, 25, 37.5, 50, 62.5, 75, 87.5],
    #            ymin=[0, 0, 0, 0, 0, 0, 0],
    #            ymax=[12.5, 25, 37.5, 50, 37.5, 25, 12.5],
    #            colors="red",
    #            linestyles="dotted")
    # plt.show()
