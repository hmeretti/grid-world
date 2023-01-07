# Exploring Agents Notebooks

Here we have some notebooks for the exploring agents. They will all be focused on solving the maze problem
under very similar conditions

They are listed below in what I believe is the natural order for reading then.


---

### monte carlo

Monte carlo is in some sense the most straight forward(maybe even naive) way of solving the problem; 
it is also not particularly efficient.

In this notebook we go through a step by step implementation of the first visit monte carlo algorithm.

---

### sarsa

Sarsa is an on-policy algorithm for solving the general reinforcement learning problem. It is pretty 
interesting and quite good at what it does.

Unlike for monte carlo, this notebook doesn't focus on the implementation(this is done at the code base
and is rather simple), but rather on exploring the performance of this algorithm and its parameters.

---

### q-learning

The big difference from q-learning and Sarsa is that this is an off-policy algorithm; which means it is
trying to find the greedy policy, even if it is not optimal for the agent itself.

Like with Sarsa we will focus exploring the performance of this algorithm and its parameters.

---

### lambda-Sarsa

This is a variation on the Sarsa algorithm, which uses eligibility traces to make multiple updates
on the q-function at every step. This is a way of improving "data-efficiency" for the Sarsa algorithm
and this notebook focus on that.

---

### lambda-q

Similar to lambda-Sarsa we add eligibility traces for the q-learning algorithm; the approach we use 
is also called the Watkin's Q(lambda), and may not be the optimal way of doing this, but it is rather
simple.

---

### q-explorer

This is an algorithm focused on the problem at hand. It is similar to q-learning, but we use a smarter
exploration policy to discard actions that can't possibly be good; which leads to significant gains in 
performance.

Even though the idea of limiting the State Action space we are looking at can be quite general, the 
way we are doing it here is very specific. Because of that this agent wouldn't work on stochastic settings.

---

### odp

This is an even more focused agent. It is designed to solve the maze problem as fast as I could imagine.

The agent creates a map of the world, based on his explorations and optimistic assumptions, and uses 
dynamic programming to plan efficient routes. This agent is guaranteed to find the greedy path and stick
to it; and it does so in an efficient way(but it is computationally expensive).

Although there is some generality to the idea here, the agent is very focused on this problem, and is
also unable to deal with stochastic settings
