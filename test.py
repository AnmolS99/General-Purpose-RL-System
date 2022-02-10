import matplotlib.pyplot as plt
import numpy as np

x = 1

l = []

num_episodes = 25000
for i in range(1, num_episodes):
    if i % (num_episodes // 100) == 0:
        x = x * 0.95
    l.append(x)

plt.plot(l)
plt.show()