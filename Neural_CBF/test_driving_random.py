"""Test script for DrivingContinuousRandom environment."""
# from imp import reload

import posggym
import time
import os
from posggym.envs.continuous.driving_continuous import ExponentialSensorModel
from train_ncbf_new import NCBFTrainer, CBFModel
from torch.utils.data import random_split, DataLoader
from trajectory_collection_parallel import build_dataset_from_env
import random
from dataset_collection import TrajectoryDataset, InputNormaliser
from unicycle import *


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    train = False
    load_dir = "./Neural_CBF/cbf_model_epoch_10.pt"
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_kwargs = {
        "id": "DrivingContinuousRandom-v0",
        "render_mode": "human",
        "obstacle_density": 0.2,
        "obstacle_radius_range": (0.4, 0.8),
        "random_seed": SEED,
        "sensor_model": ExponentialSensorModel(beta=1.0),
        "n_sensors": 32,
        "obs_dist": 5,
        "num_agents": 1,
    }
    env = posggym.make(**env_kwargs)

    print(f"Environment created: {env}")
    print(f"Agents: {env.agents}")
    print(f"Action spaces: {env.action_spaces}")
    print(f"Observation spaces: {env.observation_spaces}")

    # Reset the environment
    obs, info = env.reset(seed=SEED)
    print(f"Initial observation: {obs.keys()}")
    if train:
        dataset_dict = build_dataset_from_env(env_kwargs, num_traj=1e6, horizon=1000,
                                              save_path="./Neural_CBF/env_dataset_1e6_seed42.pkl",
                                              dump=True,
                                              balance_classes=True,
                                              processes=25)
        # Wrap into PyTorch Dataset
        full_dataset = TrajectoryDataset(dataset_dict)

        # Split sizes
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # Train/test split
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Wrap in DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        trainer = NCBFTrainer(
            obs_dim = env.observation_spaces["0"].shape[0],
            state_dim=5,
            control_dim=env.action_spaces["0"].shape[0],
            training_data=train_loader,
            test_data=test_loader,
            U_bounds=([-0.25, -np.pi/4], [0.25, np.pi/4]),
            total_epoch=10,
            mu=1,
            lambda_param =0.5,
            bound_eps=0.2,
            weight_decay = 0
        )
        model, train_loss, test_loss = trainer.train()
    else:
        model = CBFModel(env.observation_spaces["0"].shape[0]+5).to(device)
        state_dict = torch.load(load_dir, map_location=device, weights_only=True)  # "cuda:0" if you like
        model.load_state_dict(state_dict)
        model.eval()

    cbf_filter = CBF_QP_Filter(model, control_dim=2, alpha=0.1)
    norm = InputNormaliser()
    obs, _ = env.reset(seed=SEED)
    last_obs = np.ones_like(obs["0"])*5
    for t in range(1000):
        nn_ipt = np.concatenate([obs["0"],env.state[0].body[:5]])
        nn_ipt = norm.normalize_nn_input(nn_ipt)
        states = np.zeros((8,))
        states[:6] = env.state[0].body
        states[6:8] = env.state[0].dest_coord
        # nn_ipt = torch.from_numpy(nn_ipt).to(device).float()
        diff_obs = norm.normalize_obs(obs["0"] - last_obs)
        diff_obs = norm._to_numpy(diff_obs)
        u_safe = cbf_filter.safe_action(states, nn_ipt, diff_obs)
        last_obs = obs["0"]
        print(f"u_safe: {u_safe} is")
        actions = {"0": u_safe}
        step_result = env.step(actions)
        obs, rewards, terminations, truncations, _, infos = step_result

        # state = env.state[0].body[:3]
        env.render()
        if terminations["0"] or truncations["0"]:
            break
        time.sleep(0.05)
    env.close()