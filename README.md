# PyTorch CEM Implementation
This repository implements the Cross Entropy Method (CEM) for control
in pytorch. 

# Usage
Clone repository somewhere, then `pip3 install -e .` to install in editable mode.
See `tests/pendulum_approximate.py` for usage with a neural network approximating
the pendulum dynamics.

# Requirements
- pytorch (>= 1.0)
- `next state <- dynamics(state, action)` function (doesn't have to be true dynamics)
    - `state` is `K x nx`, `action` is `K x nu`
- `cost <- running_cost(state, action)` function
    - `cost` is `K x 1`, state is `K x nx`, `action` is `K x nu`

# Features
- Parallel/batch pytorch implementation for accelerated sampling
- Control bounds

# Related projects
- [pytorch MPPI](https://github.com/LemonPi/pytorch_mppi) - an alternative MPC method with similar API as this project
(faster and works better in general than CEM)