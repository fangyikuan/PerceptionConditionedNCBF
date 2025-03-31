"""Test script for DrivingContinuousRandom environment."""

import posggym
import numpy as np
import time
import pickle
from train_ncbf_new import NCBFTrainer
from tqdm import trange
def generate_random_traj(env, num_traj, horizon=100, sleep_time=0.0, render=False):
    """
    Generate random trajectories using the environment's step function.

    Args:
        env: The initialized posggym environment.
        num_traj (int): Number of trajectories to generate.
        horizon (int): Number of timesteps per trajectory.
        sleep_time (float): Delay between steps for visualization.
        render (bool): Whether to render the environment.

    Returns:
        trajectories (list): A list of trajectory dictionaries, each containing
                             'observations', 'actions', 'rewards', 'infos'.
    """
    agent_id = env.agents[0]  # Assumes single-agent for now

    trajectories = []

    for traj_idx in trange(num_traj):
        obs, info = env.reset(seed=None)  # Can pass specific seed for reproducibility
        trajectory = {
            "observations": [obs[agent_id]],
            "actions": [],
            "rewards": [],
            "states": [],
            "infos": [info]
        }

        for t in range(horizon):
            action = env.action_spaces[agent_id].sample()
            actions = {agent_id: action}
            state = env.state[agent_id]["body"]
            step_result = env.step(actions)
            if len(step_result) == 6:
                next_obs, rewards, terminations, truncations, _, infos = step_result
            else:
                next_obs, rewards, terminations, truncations, infos = step_result

            trajectory["actions"].append(action)
            trajectory["rewards"].append(rewards[agent_id])
            trajectory["observations"].append(next_obs[agent_id])
            trajectory["states"].append(state)
            trajectory["infos"].append(infos[agent_id])

            if render:
                env.render()
                time.sleep(sleep_time)

            if terminations[agent_id] or truncations[agent_id]:
                break

        trajectories.append(trajectory)

    return trajectories


def build_dataset_from_env(env, num_traj, horizon, n_ignore=50):
    """Build dataset using posggym environment and random trajectories."""
    from collections import deque

    agent_id = env.agents[0]
    Xrefs = []
    Urefs = []

    # Generate trajectories
    trajectories = generate_random_traj(env, num_traj=num_traj, horizon=horizon, render=False)

    for traj in trajectories:
        obs = traj["observations"]
        acts = traj["actions"]

        if len(acts) < n_ignore + 1:
            continue

        Xrefs.append(obs)
        Urefs.append(acts)

    data = []
    for xref, uref in zip(Xrefs, Urefs):
        for i in range(len(uref) - n_ignore):
            data.append([xref[i], uref[i], [True]])  # Safe data

    # Generate unsafe points from outside observation bounds (approximate)
    n_safe = int(np.floor(len(data) * 0.8))
    obs_space = env.observation_spaces[agent_id]
    act_space = env.action_spaces[agent_id]

    for _ in range(n_safe):
        # Unsafe state: sample outside the space bounds
        obs_low, obs_high = obs_space.low, obs_space.high
        unsafe_state = np.random.uniform(low=obs_low - 1.0, high=obs_high + 1.0)
        while np.all((unsafe_state >= obs_low) & (unsafe_state <= obs_high)):
            unsafe_state = np.random.uniform(low=obs_low - 1.0, high=obs_high + 1.0)

        unsafe_action = act_space.sample()
        data.append([unsafe_state, unsafe_action, [False]])

    # Save dataset
    with open("env_training_data.pkl", "wb") as f:
        pickle.dump(data[:-1000], f)

    with open("env_test_data.pkl", "wb") as f:
        pickle.dump(data[-1000:], f)

    print(f"Saved {len(data[:-1000])} training samples and {len(data[-1000:])} test samples")
    return data[:-1000], data[-1000:]
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
raw_training_data, raw_test_data = build_dataset_from_env(env, num_traj=20, horizon=100)
trainer = NCBFTrainer(
    state_dim=37,
    control_dim=2,
    training_data=raw_training_data,
    test_data=raw_test_data,
    U_bounds=([-1, -1], [1, 1])
)
model, train_loss, test_loss = trainer.train()
# Run a few random steps
# for i in range(100):
#     # Sample random actions for all agents
#     actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
#
#     # Step the environment
#     step_result = env.step(actions)
#
#     # Unpack the step result based on its structure
#     if isinstance(step_result, tuple) and len(step_result) == 6:
#         # Format: (obs, rewards, terminations, truncations, done, infos)
#         next_obs, rewards, terminations, truncations, _, infos = step_result
#     else:
#         # Standard format: (obs, rewards, terminations, truncations, infos)
#         next_obs, rewards, terminations, truncations, infos = step_result
#
#     # Print some information (only every 10 steps to avoid too much output)
#     if i % 10 == 0:
#         print(f"\nStep {i}:")
#         print(f"Rewards: {rewards}")
#         print(f"Terminations: {terminations}")
#         print(f"Truncations: {truncations}")
#
#     # Slow down the rendering
#     env.render()
#     time.sleep(0.1)
#
#     # Check if all agents are done
#     if all(terminations.values()) or all(truncations.values()):
#         print("\nEpisode finished, resetting environment")
#         obs, info = env.reset()

# Close the environment
env.close()