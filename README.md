# EEE4022F Robot Navigation with PPO

This repository contains a simulation framework for robot navigation using Proximal Policy Optimisation (PPO) reinforcement learning. The project enables a differential-drive robot to learn navigation strategies across a variety of map layouts, using simulated sensor data and motor voltages. The system is designed for research and educational purposes, with a focus on modularity, extensibility, and clear visualisation.

## Features

- **Custom Robot Simulation:** Models a differential-drive robot with realistic distance and goal sensors, motor physics, and collision detection.
- **Multiple Map Types:** Provides several environment layouts to evaluate navigation and control strategies.
- **PPO Agent:** Implements an enhanced PPO agent with actor-critic neural networks using PyTorch.
- **Training and Evaluation:** Includes routines for agent training, evaluation, model saving/loading, and reward logging.
- **Visualisation:** Utilises Pygame for real-time visualisation of the robot and its environment.
- **Trajectory Demonstration:** Offers a demonstration mode to visualise basic robot motions and their resulting trajectories.

## Environment Map Types

The simulation supports several map types, each designed to test specific aspects of navigation:

- **Straight Corridor:** A wide, unobstructed corridor for basic navigation tasks.
- **Narrow Corridor:** A tighter corridor requiring precise control to avoid collisions.
- **T-Junction:** A T-shaped corridor to evaluate the robot's ability to handle junctions and make directional decisions.
- **Physics Arena:** A large, open area intended for testing the fidelity of the robot's physical model and control algorithms, featuring a small goal in one corner.

All map types are defined in [`robot.py`](robot.py) and may be customised or extended as required.

## Directory Structure

```
EEE4022F/
│
├── ppo_robot_models/         # Saved PPO models and training logs
├── robot.py                  # Robot simulation, environment, and map layouts
├── ppo_agent.py              # PPO agent, training, evaluation, and reward shaping
├── README.md                 # Project documentation
└── (other scripts or data as needed)
```

## Class Overview

- **RobotEnv (robot.py):**  
  Simulates the robot and its environment, including map generation, robot physics, sensor simulation, collision detection, and rendering. Also provides a demonstration mode for visualising basic robot trajectories.

- **EnhancedPPOAgent (ppo_agent.py):**  
  Implements the PPO reinforcement learning algorithm using PyTorch. Handles policy and value networks, action selection, experience storage, and learning updates. Includes routines for training, evaluation, reward shaping, and model persistence.

## Requirements

- Python 3.11.11
- [PyTorch](https://pytorch.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Pygame](https://www.pygame.org/)
- NumPy
- Matplotlib

Install dependencies with:

```sh
pip install torch gymnasium pygame numpy matplotlib
```

## Usage

### Training

To train the PPO agent, execute:

```sh
python ppo_agent.py
```

Training statistics and models will be saved in the `ppo_robot_models` directory.

### Evaluation

By default, the script loads the best saved model and evaluates it on all map types. Results are printed to the console.

### Visualisation

Set `render=True` in the training or evaluation functions to enable real-time visualisation of the robot.

### Demonstrating Robot Trajectories

To demonstrate basic robot motions (forward, backward, turning, arching turns) and visualise their trajectories, run:

```sh
python robot.py
```

This will display the robot performing each motion in the Physics Arena, drawing the path taken for each.

## Customisation

- **Maps:** Modify or add new map types in [`robot.py`](robot.py) (`MapType` and `create_map`).
- **Rewards:** Adjust reward shaping in `EnhancedRobotEnv` in [`ppo_agent.py`](ppo_agent.py).
- **Agent Hyperparameters:** Tune PPO parameters in the `EnhancedPPOAgent` constructor.

## Example Results

After training, the script prints average rewards and success rates for each environment.

---

**Author:**  
Piwani Nkomo

**Note:**  
This code is intended for simulation and research purposes only.