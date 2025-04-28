"""Test script for DrivingContinuousRandom environment."""
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
# --------------------------------------------------------------------------- #
#  CLI ARGUMENTS
# --------------------------------------------------------------------------- #
import argparse, pathlib

def positive_int(x):    # sanity-checking
    x = int(x)
    if x <= 0:
        raise argparse.ArgumentTypeError("value must be >0")
    return x

parser = argparse.ArgumentParser(
        description="Neural-CBF demo / trainer for DrivingContinuousRandom-v0")

# --- high-level mode -------------------------------------------------------- #
parser.add_argument("--train",            action="store_true",
                    help="train a new CBF (otherwise loads checkpoint)")
parser.add_argument("--deterministic",    action="store_true",
                    help="set seed to RNG to create deterministic environment")
parser.add_argument("--seed",             type=int, default=42,
                    help="global RNG seed")
parser.add_argument("--verbose",          type=int, default=1,
                    help="Level of logging info output, 0 for silence mode; 1 for detailed mode")
# --- env / sensor ----------------------------------------------------------- #
parser.add_argument("--beta",             type=float, default=1.0,
                    help="β for ExponentialSensorModel")
parser.add_argument("--obstacle_density", type=float, default=0.2,
                    help="obstacle_density in 2D world")
parser.add_argument("--n_sensors",        type=positive_int, default=32,
                    help="# radial range sensors")
parser.add_argument("--obs_dist",         type=float, default=5.0,
                    help="sensor max distance")
parser.add_argument("--num_agents",       type=positive_int, default=1,
                    help="# vehicles in the world")

# --- data collection -------------------------------------------------------- #
parser.add_argument("--num_traj",         type=positive_int, default=int(1e6),
                    help="# transitions to collect")
parser.add_argument("--horizon",          type=positive_int, default=1000,
                    help="steps per rollout for collection")
parser.add_argument("--processes",        type=positive_int, default=25,
                    help="# parallel worker processes")

parser.add_argument("--dataset_reload_dir", type=str, default=None,
                    help="path to .pkl dataset to *load* instead of collecting")
parser.add_argument("--dataset_save_dir",  type=str,
                    default="./Neural_CBF/env_dataset.pkl",
                    help="where to save the collected dataset")
parser.add_argument("--dump",             action="store_true",
                    help="pickle the collected dataset")
parser.add_argument("--balance_class",    action="store_true",
                    help="balance safe / unsafe samples during collection")
parser.add_argument("--training_sample_portion", type=float, default=0.8,
                    help="Ratio of training samples / dataset, the rest will be testing samples")
# --- training --------------------------------------------------------------- #
parser.add_argument("--total_epoch",      type=positive_int, default=10)
parser.add_argument("--batch_size",       type=positive_int, default=512)
parser.add_argument("--mu",               type=float, default=1.0)
parser.add_argument("--lambda_param",     type=float, default=0.5)
parser.add_argument("--bound_eps",        type=float, default=0.2)
parser.add_argument("--weight_decay",     type=float, default=0.0)
parser.add_argument("--alpha",            type=float, default=0.01,
                    help="α in CBF condition dot{h}+αh ≥ 0")
# --- Roll-out Phrase Control ------------------------------------------------ #
parser.add_argument("--rollout_steps",    type=positive_int, default=1000,
                    help="Number of Steps in rollout")
# --- checkpoint paths ------------------------------------------------------- #
parser.add_argument("--nn_reload_directory", type=str,
                    default="./Neural_CBF/cbf_model_dtm_10_1e6.pt",
                    help="CBF weights to load when --train is *not* set")

args = parser.parse_args()
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    state_dim = 5

    if args.deterministic:
        SEED = args.seed
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
        "obstacle_density": args.obstacle_density,
        "obstacle_radius_range": (0.4, 0.8),
        "random_seed": args.seed if args.deterministic else None,
        "sensor_model": ExponentialSensorModel(beta=args.beta),
        "n_sensors": args.n_sensors,
        "obs_dist": args.obs_dist,
        "num_agents": args.num_agents,
    }
    env = posggym.make(**env_kwargs)

    print(f"Environment created: {env}")
    print(f"Agents: {env.agents}")
    print(f"Action spaces: {env.action_spaces}")
    print(f"Observation spaces: {env.observation_spaces}")

    # Reset the environment
    obs, info = env.reset(seed=args.seed if args.deterministic else None)
    print(f"Initial observation: {obs.keys()}")
    if args.train:
        dataset_dict = build_dataset_from_env(env_kwargs, num_traj=args.num_traj, horizon=args.horizon,
                                              reload_directory=args.dataset_reload_dir,
                                              save_path=args.dataset_save_dir,
                                              dump=args.dump,
                                              balance_classes=args.balance_class,
                                              processes=args.processes)
        # Wrap into PyTorch Dataset
        full_dataset = TrajectoryDataset(dataset_dict)

        # Split sizes
        train_size = int(args.training_sample_portion* len(full_dataset))
        test_size = len(full_dataset) - train_size

        # Train/test split
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Wrap in DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        trainer = NCBFTrainer(
            obs_dim = env.observation_spaces["0"].shape[0],
            state_dim=state_dim,
            control_dim=env.action_spaces["0"].shape[0],
            training_data=train_loader,
            test_data=test_loader,
            U_bounds=([-0.0, -np.pi/4], [0.25, np.pi/4]),
            total_epoch=args.total_epoch,
            mu=args.mu,
            lambda_param =args.lambda_param,
            bound_eps=args.bound_eps,
            weight_decay = args.weight_decay
        )
        model, train_loss, test_loss = trainer.train()
    else:
        model = CBFModel(env.observation_spaces["0"].shape[0] + state_dim).to(device)
        state_dict = torch.load(args.nn_reload_directory, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

    cbf_filter = CBF_QP_Filter(model, control_dim=env.action_spaces["0"].shape[0], alpha=args.alpha)
    norm = InputNormaliser()
    obs, _ = env.reset(seed=args.seed if args.deterministic else None)
    last_obs = np.ones_like(obs["0"]) * args.obs_dist
    for t in range(args.rollout_steps):
        nn_ipt = np.concatenate([obs["0"],env.state[0].body[:5]])
        nn_ipt = norm.normalize_nn_input(nn_ipt, device=device)
        states = np.zeros((8,))
        states[:6] = env.state[0].body
        states[6:8] = env.state[0].dest_coord
        # nn_ipt = torch.from_numpy(nn_ipt).to(device).float()
        diff_obs = norm.normalize_obs(obs["0"] - last_obs)
        diff_obs = norm._to_numpy(diff_obs)
        u_safe = cbf_filter.safe_action(states, nn_ipt, diff_obs)
        last_obs = obs["0"]
        if args.verbose == 1:
            print(f"u_safe: {u_safe} is")
        actions = {"0": u_safe}
        step_result = env.step(actions)
        obs, rewards, terminations, truncations, _, infos = step_result

        env.render()
        if terminations["0"] or truncations["0"]:
            break
        time.sleep(0.05)
    env.close()