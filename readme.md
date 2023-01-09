## Intro

![](https://github.com/hmeretti/grid-world/blob/main/q_vs_q_world_01.gif)


If you are wondering "wtf is that?"; well it is a @ chasing a # in a grid world. Those are, in fact, two reinforcement
learning agents trained in an adversarial task - the @ manages to win by using diagonal moves, 
which are not available for the #.

The project contains a simple grid world simulation and defines two distinct problems:

* the maze: where an agent is trying to navigate the world and reach a specific destination
* the tag game: where two agents play against each other, one trying to catch and the other trying to run.

We explore methods for solving these problems, such as dynamic programming and reinforcement learning. 

This is intended as a study on how to implement such methods from scratch in a controlled environment,
and to provide a simple experimentation setup. It is by no means intended as a general efficient implementation.

Every thing here is heavily based on the book 
["Reinforcement Learning: An Introduction"](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) 
from Richard S. Sutton and Andrew G. Barto(which is awsome).

For a quick overview of the problem and notation I recommend checking the [intro notebook](https://github.com/hmeretti/grid-world/blob/main/intro.ipynb) 
on the root folder of this project. Some interesting notebooks about the maze problem can be found [here](https://github.com/hmeretti/grid-world/tree/main/notebooks/exploring_agents); and [here](https://github.com/hmeretti/grid-world/tree/main/notebooks/tag) you will find notebooks about the tag game.

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
│   └── training
├── grid_world
│   └── visualization
│       └── animation_scripts
├── notebooks
│   ├── dynamic_programing
│   ├── exploring_agents
│   ├── tag
│   └── utils
├── persistence
│   ├── agents
│   └── training_scripts
├── policies
└── utils

```

The code under `grid_world` contains a definition of the world including its dynamics
and useful signals it may return. 

Agents that learn policies by exploring the world can be found at `exploring_agents`. There are two
types of agent: the generic ones which can be used to solve any RL problem(including the both problems we have here of 
course), and the grid world agents which make use of stuff specific to this world, 
and are designed for the maze problem. Under `exploring_agents\training` we have ways of training 
these agents by making then interact with a world; 
there are functions to train then to solve the maze problem as well as the tag problem.

The DP  implementation is at `dynamic_programing`. It is also generic enough to work in any problem
with a single agent.

At `notebooks\exploring_agents` you will find explanations on how the implemented agents
work and some exploration of they parameters. At `notebooks\tag` there is some exploration on the tag
problem.

Under `notebooks\dynamic_progaming` you can find examples and details to solve the maze problem using DP
(this is outdated in relation to the rest of the project though).

You can find some scripts to generate visualizations of the agent running throw the world in 
`grid_world/visualization/animation_scripts`. They are pretty simple, and can be edited quite easily. 
Some of them require using pre-trained agents, which you can create running the corresponding notebooks
(artifacts are not being committed in the project). 
Visualizations work fine on linux, but I don't know if they will work on Windows or Mac.

`Abstractions` contains type definitions and abstract classes to make the interface we are using clear.

## Setting local environment

If you have pyenv installed just run the following inside the base dir:

```
make environment
make requirements
```

This will create a local environment, set as the default for this dir, and install all requirements.

If not you can create a `python 3.11.1` environment and install the dependencies under `requirements.txt`

## Tests

There are some tests in the project, although not as comprehensive as I'd like. Some of them are stochastic
in nature, which means they could occasionally fail(didn't happen to me so far). You can run then with

``make test ``

Apart from test code style is validated with `black`. You can apply it by running

``make black ``

Validate everything(tests and style) with

``make validate ``

passing this is required for merging pull requests.
