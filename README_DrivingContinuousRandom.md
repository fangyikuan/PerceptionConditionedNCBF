# DrivingContinuousRandom Environment

This is a custom environment for the POSGGym library that extends the existing DrivingContinuous environment by adding random circular obstacles instead of the hard-coded block obstacles.

## Overview

The DrivingContinuousRandom environment is based on the DrivingContinuous-v0 environment but allows for more flexible obstacle configurations. Instead of using predefined block obstacles, this environment generates random circular obstacles based on specified parameters.

## Features

- Random circular obstacles with configurable size range and density
- All the features of the original DrivingContinuous environment
- Customizable obstacle field for different difficulty levels
- Reproducible obstacle generation with random seed

## Installation

The environment is included in the POSGGym package. No additional installation is required if you have POSGGym installed.

## Usage

```python
import posggym

# Create the environment with default parameters
env = posggym.make("DrivingContinuousRandom-v0")

# Create the environment with custom parameters
env = posggym.make(
    "DrivingContinuousRandom-v0",
    world="14x14Empty",  # Use the empty world layout
    num_agents=2,  # Number of agents
    obs_dist=5.0,  # Observation distance
    n_sensors=16,  # Number of sensors
    obstacle_radius_range=(0.3, 0.7),  # Min and max obstacle radius
    obstacle_density=0.1,  # Density of obstacles (0.0 to 1.0)
    random_seed=42,  # Random seed for reproducibility
    render_mode="human",  # Render mode (optional)
)

# Reset the environment
obs, info = env.reset()

# Run a simulation with random actions
for _ in range(100):
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # Check if episode is done
    if all(terminations.values()) or all(truncations.values()):
        obs, info = env.reset()

# Close the environment
env.close()
```

## Parameters

- `world` (str or RandomDrivingWorld): The world layout to use. Default is "14x14Empty".
- `num_agents` (int): The number of agents in the environment. Default is 2.
- `obs_dist` (float): The sensor observation distance. Default is 5.0.
- `n_sensors` (int): The number of sensor lines eminating from the agent. Default is 16.
- `obstacle_radius_range` (tuple): A tuple (min_radius, max_radius) specifying the range of possible obstacle radii. Default is (0.3, 0.7).
- `obstacle_density` (float): A float between 0 and 1 specifying the density of obstacles in the environment. Default is 0.1. Higher values mean more obstacles.
- `random_seed` (int, optional): A seed for the random number generator to ensure reproducible obstacle generation. Default is None.
- `render_mode` (str, optional): The render mode to use. Can be "human" or "rgb_array". Default is None.

## Available World Layouts

- `14x14Empty`: An empty 14x14 world with start positions at the corners and destinations in the middle.

You can also use any of the standard world layouts from the original DrivingContinuous environment:

- `6x6Intersection`
- `7x7Blocks`
- `7x7CrissCross`
- `7x7RoundAbout`
- `14x14Blocks`
- `14x14CrissCross`
- `14x14RoundAbout`

## Example

Here's a simple example of using the environment:

```python
import posggym
import time

# Create the environment
env = posggym.make(
    "DrivingContinuousRandom-v0",
    render_mode="human",
    obstacle_density=0.2,  # Increase density for more obstacles
    obstacle_radius_range=(0.2, 0.8),  # Vary the size of obstacles
    random_seed=42,  # Set seed for reproducibility
)

# Reset the environment
obs, info = env.reset()

# Run a few random steps
for _ in range(100):
    # Sample random actions for all agents
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    
    # Step the environment
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # Slow down the rendering
    time.sleep(0.1)
    
    # Check if all agents are done
    if all(terminations.values()) or all(truncations.values()):
        obs, info = env.reset()

# Close the environment
env.close()
```

## Extending the Environment

You can create your own custom world layouts by defining a new world string in the `SUPPORTED_RANDOM_WORLDS` dictionary in the `driving_continuous_random.py` file.

## License

This environment is released under the same license as the POSGGym library.