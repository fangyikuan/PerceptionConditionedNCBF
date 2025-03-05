"""Test script for DrivingContinuousRandom environment."""

import posggym
import numpy as np
import time

# Create the environment
env = posggym.make(
    "DrivingContinuousRandom-v0",
    render_mode="human",
    obstacle_density=0.2,  # Increase density for more obstacles
    obstacle_radius_range=(0.2, 0.8),  # Vary the size of obstacles
    random_seed=42,  # Set seed for reproducibility
)

print(f"Environment created: {env}")
print(f"Agents: {env.agents}")
print(f"Action spaces: {env.action_spaces}")
print(f"Observation spaces: {env.observation_spaces}")

# Reset the environment
obs, info = env.reset()
print(f"Initial observation: {obs.keys()}")

# Run a few random steps
for i in range(100):
    # Sample random actions for all agents
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    
    # Step the environment
    step_result = env.step(actions)
    
    # Unpack the step result based on its structure
    if isinstance(step_result, tuple) and len(step_result) == 6:
        # Format: (obs, rewards, terminations, truncations, done, infos)
        next_obs, rewards, terminations, truncations, _, infos = step_result
    else:
        # Standard format: (obs, rewards, terminations, truncations, infos)
        next_obs, rewards, terminations, truncations, infos = step_result
    
    # Print some information (only every 10 steps to avoid too much output)
    if i % 10 == 0:
        print(f"\nStep {i}:")
        print(f"Rewards: {rewards}")
        print(f"Terminations: {terminations}")
        print(f"Truncations: {truncations}")
    
    # Slow down the rendering
    time.sleep(0.1)
    
    # Check if all agents are done
    if all(terminations.values()) or all(truncations.values()):
        print("\nEpisode finished, resetting environment")
        obs, info = env.reset()

# Close the environment
env.close()