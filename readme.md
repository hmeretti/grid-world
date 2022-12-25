# WIP

Still working on lots of functionalities; things are changing a lot. Some agents are probably currently not working.

## Intro

This project contains a simple grid world simulation and methods for solving the problem, such as dynamic programming and
reinforcement learnig. This is intended as a study on how to implement such methods from scratch in a controlled environment,
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

```bash
├── dynamic_programing
├── grid_world
│   ├── agents
│   ├── utils
│   └── visualization
│       └── animation_scripts
├── notebooks
│   ├── dynamic_programing
│   ├── experiments
│   └── reinforcement_learning
│       └── agents
└── utils
```

The main code is under `grid_world`, where we have a definition of the world including its dynamics
and useful signals it may return. The RL agents can be found at `grid_world/agents` while the DP
implementation is at `dynamic_programing`. This is in a stand-alone directory because, contrary to the
RL agents, this implementation should be general enough for any problem, as long as it is implemented
in way conforming to the specified type aliases.

You can find some scripts to generate visualizations of the agent running throw the world in 
`grid_world/visualization/animation_scripts`. They are pretty simple, and can be edited quite easily. 
Don't know if they work on Windows or Mac.

Under `notebooks\dynamic_progaming` you can find ways to solve our problem using DP. While at 
`notebooks\reinforcement_learning\agents` you will find explanations on how the implemented agents
work. Under `notebooks\experiments` we are supposed to have some experiments exploring the differences
between these methods.