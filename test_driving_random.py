"""Test script for DrivingContinuousRandom environment."""
from imp import reload

import posggym
import numpy as np
import time
from posggym.envs.continuous.driving_continuous import ExponentialSensorModel
import pickle
from train_ncbf_new import NCBFTrainer
from tqdm import trange
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import os
from collections import Counter
class TrajectoryDataset(Dataset):
    def __init__(self, data_dict):
        self.observations = torch.tensor(data_dict["observation"], dtype=torch.float32)
        self.obs_diffs = torch.tensor(data_dict["observation_diff"], dtype=torch.float32)
        self.actions = torch.tensor(data_dict["action"], dtype=torch.float32)
        self.states = torch.tensor(data_dict["states"], dtype=torch.float32)
        self.labels = torch.tensor(data_dict["label"], dtype=torch.float32)  # can be float or long for BCE/CrossEntropy

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "observation": self.observations[idx],
            "observation_diff": self.obs_diffs[idx],
            "action": self.actions[idx],
            "state": self.states[idx],
            "label": self.labels[idx]
        }
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
    traj_lengths = []
    env._max_episode_steps = horizon
    for traj_idx in trange(num_traj):
        obs, info = env.reset(seed=None)  # Can pass specific seed for reproducibility
        trajectory = {
            "observations": [obs[agent_id]],
            "observation_diff": [np.zeros(obs[agent_id].shape)],
            "actions": [],
            "rewards": [],
            "states": [],
            "safe":[],
            "infos": [info]
        }

        for t in range(horizon):
            action = env.action_spaces[agent_id].sample()
            actions = {agent_id: action}
            state = env.state[int(agent_id)].body[:3]

            step_result = env.step(actions)
            if len(step_result) == 6:
                next_obs, rewards, terminations, truncations, _, infos = step_result
            else:
                next_obs, rewards, terminations, truncations, infos = step_result
            obs_diff = next_obs[agent_id] - trajectory["observations"][-1]
            crashed = env.state[int(agent_id)].status[1]
            if crashed == 0:
                is_safe = True
            else:
                is_safe = False
            trajectory["actions"].append(action)
            trajectory["rewards"].append(rewards[agent_id])
            trajectory["observations"].append(next_obs[agent_id])
            trajectory["observation_diff"].append(obs_diff)
            trajectory["states"].append(state)
            trajectory["safe"].append(is_safe)
            trajectory["infos"].append(infos[agent_id])


            if render:
                env.render()
                time.sleep(sleep_time)

            if terminations[agent_id] or truncations[agent_id]:
                break

        trajectories.append(trajectory)
        traj_lengths.append(len(trajectory["actions"]))
    print("\n📊 Trajectory Length Distribution:")
    length_counts = Counter(traj_lengths)
    for length in sorted(length_counts):
        bar = '█' * (length_counts[length] * 50 // num_traj)  # scale bar length
        print(f"Length {length:3d}: {length_counts[length]:4d} | {bar}")

    return trajectories


def build_dataset_from_env(env, num_traj, horizon, n_ignore=50, render=False, dump=True,
                           reload_directory=None, balance_classes=False, save_path="env_dataset.pkl"):
    """Build dataset using posggym environment and random trajectories."""

    env.spec.max_episode_steps = horizon

    if reload_directory is not None:
        with open(reload_directory, "rb") as f:
            dataset_dict = pickle.load(f)
    else:
        trajectories = generate_random_traj(env, num_traj=num_traj, horizon=horizon, render=render)

        obs_data = []
        obs_diff_data = []
        action_data = []
        state_data = []
        safe_labels = []

        for traj in trajectories:
            obs = traj["observations"]
            obs_diff = traj["observation_diff"]
            acts = traj["actions"]
            states = traj["states"]
            safe_flags = traj["safe"]

            traj_len = len(acts)

            if traj_len > n_ignore + 1:
                end_idx = traj_len - (n_ignore + 1)
                obs_data.extend(obs[:end_idx])
                obs_diff_data.extend(obs_diff[:end_idx])
                action_data.extend(acts[:end_idx])
                state_data.extend(states[:end_idx])
                safe_labels.extend([True] * end_idx)
            elif traj_len <= n_ignore and safe_flags[-1] == False:
                obs_data.append(obs[-1])
                obs_diff_data.append(obs_diff[-1])
                action_data.append(acts[-1])
                state_data.append(states[-1])
                safe_labels.append(False)

        # Convert to NumPy arrays before indexing
        obs_data = np.array(obs_data)
        obs_diff_data = np.array(obs_diff_data)
        action_data = np.array(action_data)
        state_data = np.array(state_data)
        safe_labels = np.array(safe_labels, dtype=bool)

        if balance_classes:
            safe_indices = np.where(safe_labels == True)[0]
            unsafe_indices = np.where(safe_labels == False)[0]
            min_class_size = min(len(safe_indices), len(unsafe_indices))

            np.random.shuffle(safe_indices)
            np.random.shuffle(unsafe_indices)

            keep_indices = np.concatenate([safe_indices[:min_class_size], unsafe_indices[:min_class_size]])
            np.random.shuffle(keep_indices)

            obs_data = obs_data[keep_indices]
            obs_diff_data = obs_diff_data[keep_indices]
            action_data = action_data[keep_indices]
            state_data = state_data[keep_indices]
            safe_labels = safe_labels[keep_indices]

            print(f"Balanced dataset to {len(keep_indices)} total samples ({min_class_size} per class).")
        else:
            print(f"Unbalanced dataset with {np.sum(safe_labels)} safe and {np.sum(~safe_labels)} unsafe samples.")

        dataset_dict = {
            "observation": obs_data,
            "observation_diff": obs_diff_data,
            "action": action_data,
            "states": state_data,
            "label": safe_labels
        }

    print(f"Final dataset size: {len(dataset_dict['observation'])} samples.")

    # Save to disk if requested
    if dump:
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(dataset_dict, f)
        print(f"Dataset saved to: {save_path}")

    return dataset_dict
# Create the environment



sensor_model = ExponentialSensorModel(beta=1.0)
env = posggym.make(
    "DrivingContinuousRandom-v0",
    render_mode="human",
    obstacle_density=0.2,  # Increase density for more obstacles
    obstacle_radius_range=(0.4, 0.8),  # Larger obstacles
    random_seed=42,  # Set seed for reproducibility
    sensor_model=sensor_model,  # Use our exponential sensor model
    n_sensors=32,
    obs_dist=5,
    num_agents=1,
)

print(f"Environment created: {env}")
print(f"Agents: {env.agents}")
print(f"Action spaces: {env.action_spaces}")
print(f"Observation spaces: {env.observation_spaces}")

# Reset the environment
obs, info = env.reset()
print(f"Initial observation: {obs.keys()}")
dataset_dict = build_dataset_from_env(env, num_traj=10000, horizon=1000,
                                      reload_directory="./env_dataset.pkl",
                                      dump=False)
# Wrap into PyTorch Dataset
full_dataset = TrajectoryDataset(dataset_dict)

# Split sizes
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# Train/test split
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Wrap in DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

trainer = NCBFTrainer(
    obs_dim = env.observation_spaces["0"].shape[0],
    state_dim=3,
    control_dim=env.action_spaces["0"].shape[0],
    training_data=train_loader,
    test_data=test_loader,
    U_bounds=([-1, -1], [1, 1])
)
model, train_loss, test_loss = trainer.train()
# Run a few random steps
# for i in range(200):
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
#         print(f"Info: {infos}")
#
#     # Check for collisions and print them
#     for agent_id, info in infos.items():
#         if 'outcome' in info and hasattr(info['outcome'], 'value') and info['outcome'].value == -1:
#             print(f"\nCOLLISION DETECTED at step {i} for agent {agent_id}!")
#             print(f"Reward: {rewards[agent_id]}")
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