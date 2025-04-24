import os
import time
import pickle
import numpy as np
import posggym
from collections import Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def _collect_single_traj(args):
    """
    Worker function to collect one trajectory.
    Args:
        args: tuple (env_kwargs, horizon, seed)
    Returns:
        trajectory dict and its length
    """
    try:
        env_kwargs, horizon, seed = args
        env_kwargs["render_mode"] = None
        env = posggym.make(**env_kwargs)
        obs, info = env.reset(seed=seed)
        agent_id = env.agents[0]
        env._max_episode_steps = horizon
        trajectory = {
            "observations": [obs[agent_id]],
            "observation_diff": [np.zeros_like(obs[agent_id])],
            "actions": [],
            "rewards": [],
            "states": [],
            "safe": [],
            "infos": [info[agent_id]],
        }
        action_rng = np.random.default_rng()
        for t in range(horizon):
            space = env.action_spaces[agent_id]
            low, high = space.low, space.high
            low[1] = 0 # Limit the robot to only go forward
            action = action_rng.random(size=low.shape) * (high - low) + low
            # action = env.action_spaces[agent_id].sample() # angular vel, linear acc
            actions = {agent_id: action}
            state = env.state[int(agent_id)].body[:5] # [x,y,theta,vx,vy]

            step = env.step(actions)
            if len(step) == 6:
                next_obs, rewards, terminations, truncations, _, infos = step
            else:
                next_obs, rewards, terminations, truncations, infos = step

            obs_diff = next_obs[agent_id] - trajectory["observations"][-1]
            crashed = env.state[int(agent_id)].status[1]
            is_safe = (crashed == 0)
            action = action[::-1] # linear acc,  angular vel
            trajectory["actions"].append(action)
            trajectory["rewards"].append(rewards[agent_id])
            trajectory["states"].append(state)
            trajectory["safe"].append(is_safe)
            trajectory["infos"].append(infos[agent_id])

            if terminations[agent_id] or truncations[agent_id]:
                break
            trajectory["observations"].append(next_obs[agent_id])
            trajectory["observation_diff"].append(obs_diff)

        env.close()
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)
    return trajectory, len(trajectory["actions"])


def generate_random_traj_parallel(env_kwargs, num_traj, horizon=100, processes=None):
    """
    Parallel trajectory generation across CPU cores with progress bar.
    Args:
        env_kwargs   : dict of arguments for posggym.make
        num_traj (int): total trajectories to collect
        horizon  (int): max steps per trajectory
        processes    : number of worker processes (defaults to cpu_count())
    Returns:
        List of trajectory dicts
    """
    num_traj = int(num_traj)
    if processes is None:
        processes = cpu_count()
    _seed = env_kwargs["random_seed"]
    args_list = [(env_kwargs, horizon, _seed) for _ in range(num_traj)]
    trajectories = []
    lengths = []

    with Pool(processes) as pool:
        for traj, length in tqdm(pool.imap(_collect_single_traj, args_list),
                                 total=num_traj,
                                 desc="Collecting trajectories"):
            trajectories.append(traj)
            lengths.append(length)

    print("\n📊 Trajectory Length Distribution:")
    length_counts = Counter(lengths)
    for length, count in sorted(length_counts.items()):
        bar = '█' * (count * 50 // num_traj)
        print(f"Length {length:3d}: {count:4d} | {bar}")

    return trajectories


def build_dataset_from_env(env_kwargs,
                           num_traj,
                           horizon,
                           n_ignore=50,
                           render=False,
                           dump=True,
                           reload_directory=None,
                           balance_classes=False,
                           save_path="env_dataset.pkl",
                           processes=None):
    """
    Build a dataset from random trajectories collected in parallel.

    Args:
        env_kwargs   : dict of arguments for posggym.make
        num_traj     : number of trajectories to generate
        horizon      : max timesteps per trajectory
        n_ignore     : ignore first n_ignore steps when labeling safe trajectories
        render       : whether to render env during collection (inefficient in parallel)
        dump         : whether to pickle the final dataset
        reload_directory: path to load existing dataset dict
        balance_classes: whether to balance safe/unsafe classes
        save_path    : file path to dump dataset
        processes    : number of CPU processes for parallel collection

    Returns:
        dataset_dict with keys ['observation','observation_diff','action','states','label']
    """
    if reload_directory is not None:
        with open(reload_directory, "rb") as f:
            dataset_dict = pickle.load(f)
    else:
        trajectories = generate_random_traj_parallel(env_kwargs,
                                                     num_traj=int(num_traj),
                                                     horizon=horizon,
                                                     processes=processes)

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
            L = len(acts)

            if L > n_ignore + 1:
                end_idx = L - (n_ignore + 1)
                obs_data.extend(obs[:end_idx])
                obs_diff_data.extend(obs_diff[:end_idx])
                action_data.extend(acts[:end_idx])
                state_data.extend(states[:end_idx])
                safe_labels.extend([True] * end_idx)
            elif L <= n_ignore and not safe_flags[-1]:
                obs_data.append(obs[-1])
                obs_diff_data.append(obs_diff[-1])
                action_data.append(acts[-1])
                state_data.append(states[-1])
                safe_labels.append(False)

        obs_data = np.array(obs_data)
        obs_diff_data = np.array(obs_diff_data)
        action_data = np.array(action_data)
        state_data = np.array(state_data)
        safe_labels = np.array(safe_labels, dtype=bool)

        if balance_classes:
            safe_idx = np.where(safe_labels)[0]
            unsafe_idx = np.where(~safe_labels)[0]
            m = min(len(safe_idx), len(unsafe_idx))
            np.random.shuffle(safe_idx)
            np.random.shuffle(unsafe_idx)
            keep = np.concatenate([safe_idx[:m], unsafe_idx[:m]])
            np.random.shuffle(keep)

            obs_data = obs_data[keep]
            obs_diff_data = obs_diff_data[keep]
            action_data = action_data[keep]
            state_data = state_data[keep]
            safe_labels = safe_labels[keep]

            print(f"Balanced to {len(keep)} samples ({m} per class)")
        else:
            print(f"Unbalanced: {np.sum(safe_labels)} safe, {np.sum(~safe_labels)} unsafe")

        dataset_dict = {
            "observation": obs_data,
            "observation_diff": obs_diff_data,
            "action": action_data,
            "states": state_data,
            "label": safe_labels,
        }

    print(f"Final dataset size: {len(dataset_dict['observation'])} samples.")

    if dump:
        with open(save_path, "wb") as f:
            pickle.dump(dataset_dict, f)
        print(f"Saved dataset to {save_path}")

    return dataset_dict

if __name__ == "__main__":
    from posggym.envs.continuous.driving_continuous import ExponentialSensorModel
    import multiprocessing as mp

    mp.set_start_method("fork")
    env_kwargs = {
        "id": "DrivingContinuousRandom-v0",
        "render_mode": None,
        "obstacle_density": 0.2,
        "obstacle_radius_range": (0.4, 0.8),
        "random_seed": None,
        "sensor_model": ExponentialSensorModel(beta=1.0),
        "n_sensors": 32,
        "obs_dist": 5,
        "num_agents": 1,
    }
    data = build_dataset_from_env(env_kwargs, num_traj=100, horizon=1000, processes=1)
