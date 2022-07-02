# General Purpose Reinforcment Learning System

This project was part of the Artificial Intelligence Programming (IT3105) course at NTNU spring 2022. The aim of this project was to build a general purpose Actor-Critic Model (ACM) for Reinforcement Learning, applied it to three different problems: pole-balancing, Towers of Hanoi, and the gambler.

## System Overview

The system consists of an actor that interacts with an environment (called SimWorlds). The actor stands for the core RL learning, while the environment houses everything else like maintaining the current state, provide the agent with current and successor (child) states, giving rewards based on state transitions and determining if the agent is in a end state. The actor consists of an actor-critic model. When deciding
upon the next action to take from game state s, the actor may consult the critic to get the values of all child
states of s. Through learning, the critic will have developed a thorough set of associations between states
and values (these can be represented in a table or in a neural net), and these will form the basis of the actor’s decision.


![Tower of Hanoi](/images/tohsw.gif)
