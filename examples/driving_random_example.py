"""Example script for DrivingContinuousRandom environment.

This script demonstrates the DrivingContinuousRandom environment with different
obstacle densities and sizes.
"""

import posggym
import time
import argparse


def run_environment(
    obstacle_density=0.1,
    obstacle_radius_range=(0.3, 0.7),
    random_seed=None,
    num_steps=200,
    render_mode="human",
):
    """Run the DrivingContinuousRandom environment with specified parameters."""
    print(f"\nRunning with obstacle density: {obstacle_density}, radius range: {obstacle_radius_range}")
    
    # Create the environment
    env = posggym.make(
        "DrivingContinuousRandom-v0",
        render_mode=render_mode,
        obstacle_density=obstacle_density,
        obstacle_radius_range=obstacle_radius_range,
        random_seed=random_seed,
    )
    
    # Reset the environment
    obs, info = env.reset()
    
    # Run for specified number of steps
    for i in range(num_steps):
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
        
        # Print some information (only every 50 steps to avoid too much output)
        if i % 50 == 0:
            print(f"Step {i}:")
            print(f"  Rewards: {rewards}")
            print(f"  Terminations: {terminations}")
        
        # Slow down the rendering
        if render_mode == "human":
            time.sleep(0.05)
        
        # Check if all agents are done
        if all(terminations.values()) or all(truncations.values()):
            print("\nEpisode finished, resetting environment")
            obs, info = env.reset()
    
    # Close the environment
    env.close()


def main():
    """Run the example with different obstacle configurations."""
    parser = argparse.ArgumentParser(description="DrivingContinuousRandom environment example")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps per configuration")
    args = parser.parse_args()
    
    render_mode = None if args.no_render else "human"
    
    # Run with different obstacle densities
    densities = [0.05, 0.2, 0.4]
    radius_ranges = [(0.2, 0.4), (0.3, 0.7), (0.5, 1.0)]
    
    for density in densities:
        for radius_range in radius_ranges:
            run_environment(
                obstacle_density=density,
                obstacle_radius_range=radius_range,
                random_seed=args.seed,
                num_steps=args.steps,
                render_mode=render_mode,
            )


if __name__ == "__main__":
    main()