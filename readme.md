## Intro

![](https://github.com/hmeretti/grid-world/blob/main/q_vs_q_world_01.gif)

This project contains a simple grid world simulation and methods for solving the problem, such as dynamic programming and
reinforcement learning. This is intended as a study on how to implement such methods from scratch in a controlled environment,
and to provide a simple experimentation setup. It is by no means intended as a general efficient implementation.
Every thing here is heavily based on the book ["Reinforcement Learning: An Introduction"](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) from Richard S. Sutton and Andrew G. Barto(which is really great by the way).

For a quick overview of the problem and notation I recommend checking the `intro.ipynb` notebook on the root folder of this project.

## Setting local environment

If you have pyenv installed just run the following inside the base dir:

```
make environment
make requirements
```

This will create a local environment, set as the default for this dir, and install all requirements.

## Structure

The project structure is very simple

```
├── abstractions
├── dynamic_programing
├── exploring_agents
│   ├── commons
│   ├── generic_agents
│   ├── grid_world_agents
│   │   └── commons
│   ├── policies
│   └── training
├── grid_world
│   └── visualization
│       └── animation_scripts
├── notebooks
│   ├── dynamic_programing
│   ├── experiments
│   ├── exploring_agents
│   ├── tag
│   └── utils
├── persistence
│   ├── agents
│   └── training_scripts
└── utils

```

The code under `grid_world` contains a definition of the world including its dynamics
and useful signals it may return. 

Agents that learn policies by exploring the world can be found at `exploring_agents`. There are two
types of agent: the generic ones which can be used to solve any RL problem, and the grid world agents
which make use of stuff specific to this problem. Under `exploring_agents\training` we have ways of
training these agents by making then interact with a world; there are functions to train then to solve
the maze problem as well as the tag problem.

The DP  implementation is at `dynamic_programing`. It is also generic enough to work in any problem
of this type.

Under `notebooks\dynamic_progaming` you can find ways to solve the maze problem using DP. While at 
`notebooks\exploring_agents` you will find explanations on how the implemented agents
work and some exploration of they parameters. At `notebooks\tag` there is some exploration on the tag
problem.

Under `notebooks\experiments` we are supposed to have some experiments exploring the differences between 
these methods(not much there yet, and very outdated).

You can find some scripts to generate visualizations of the agent running throw the world in 
`grid_world/visualization/animation_scripts`. They are pretty simple, and can be edited quite easily. 
Some of than require using pre-trained agents, which you can find under `persistence/training_scripts`. 
Visualizations work fine on linux, but I don't know if they will work on Windows or Mac.

`Abstractions` contains type definitions and abstract classes to make the interface we are using clear.