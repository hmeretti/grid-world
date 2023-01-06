# Tag Notebooks

Here we have some notebooks about the tag problem. They are listed below in what I believe is the 
natural order for reading then.

The notebooks referring to world 01 are pretty similar, and show that the agents are capable of solving
the problem at hand. The other ones explore some interesting things in different setups.

---

### Q world 01

This is a basic exploration of the q-learning agent in a tag problem. We use it to play against some random
agents and against itself.

---

### Sarsa world 01

Similar to the example above,  using SARSA. We highlight some differences between on-policy and 
off-policy methods.

---

### Lambda Sarsa world 01

Very similar to the example above, using lambda-SARSA. 

---

### Q empty world

Here we explore the problem of two agents in an empty world. We can see that there is a winning strategy
for one of the agents, which can change depending on the available moves. We also see that the agent
is capable of exploiting "bad" moves from his adversary; which, in this case, come from a high exploration
rate in its policy.

---

### Q large world


This may be a more challenging setting for the agents. However, they can solve it pretty well. Here we
have a more complicated maze they have to navigate, and we give Agent 1 diagonal moves, so he is capable
of catching Agent 2. It is interesting to see how well both agents can learn in this somewhat complex
world(nothing too crazy though).


