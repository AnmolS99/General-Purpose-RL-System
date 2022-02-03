import numpy as np


class PoleBalancingSimWorld():

    def __init__(self,
                 num_episodes,
                 max_steps,
                 l=0.5,
                 m_p=0.1,
                 g=9.8,
                 timestep=0.02) -> None:

        self.num_episodes = num_episodes
        self.max_steps = max_steps

        self.l = l  # Length of the pole
        self.m_p = m_p  # Mass of the pole
        self.g = g  # Gravity

        self.m_c = 1  # Mass of the cart
        self.theta = None
        self.theta_first_der = None
        self.theta_second_der = None
        self.x = None
        self.x_vel = 0
        self.x_acc = 0
        self.f = 10
        self.b = self.f
        self.theta_m = 0.21
        self.x_minus = -2.4
        self.x_plus = 2.4
        self.timestep = timestep
        self.episode_len = 300

    def begin_episode(self):
        # Centering the cart at the horizontal position
        self.x = (self.x_minus + self.x_plus) / 2

        # Randomly choosing theta (the pole angle)
        #self.theta = np.random.uniform(-self.theta_m, self.theta_m)
        self.theta = 0.01

        # Setting theta first temporal derivative to 0
        self.theta_first_der = 0

        for i in range(self.episode_len):
            self.next_state()
            print(self.get_current_state())
            if not (self.theta_in_range() and self.x_in_range()):
                print("Task failed")
                break

    def next_state(self):

        self.b = -self.f

        # Calculating all the relationships
        theta_second_numerator = self.g * np.sin(self.theta) + np.cos(
            self.theta) * ((-self.b - self.m_p * self.l *
                            (self.theta_first_der**2) * np.sin(self.theta)) /
                           (self.m_c + self.m_p))

        theta_second_denominator = self.l * ((4 / 3) -
                                             ((self.m_p *
                                               (np.cos(self.theta)**2)) /
                                              (self.m_c + self.m_p)))
        self.theta_second_der = theta_second_numerator / theta_second_denominator

        self.x_acc = (self.b + self.m_p * self.l *
                      ((self.theta_first_der**2) * np.sin(self.theta) -
                       self.theta_second_der * np.cos(self.theta))) / (
                           self.m_p + self.m_c)

        self.x = self.x + self.timestep * self.x_vel
        self.x_vel = self.x_vel + self.timestep * self.x_acc

        self.theta = self.theta + self.timestep * self.theta_first_der
        self.theta_first_der = self.theta_first_der + self.timestep * self.theta_second_der

    def get_current_state(self):
        """
        Returns current state, but with rounded values so that the number of possible states stays relatively small
        """
        return np.round(self.x, 3), np.round(self.x_vel, 3), np.round(
            self.theta, 3), np.round(self.theta_first_der, 3)

    def theta_in_range(self):
        return abs(self.theta) <= self.theta_m

    def x_in_range(self):
        return self.x > self.x_minus and self.x < self.x_plus


if __name__ == "__main__":
    pbsw = PoleBalancingSimWorld(1, 1)
    pbsw.begin_episode()