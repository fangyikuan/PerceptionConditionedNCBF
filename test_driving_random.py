"""Test script for DrivingContinuousRandom environment."""
from imp import reload

import posggym
import numpy as np
import time

from collect_data import DubinsCar
from posggym.envs.continuous.driving_continuous import ExponentialSensorModel
import pickle
from train_ncbf_new import NCBFTrainer, CBFModel
from tqdm import trange
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import os
from collections import Counter
import cvxpy as cp
train = False
load_dir = "./cbf_model_epoch_10.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

class PIDGoalController:
    """
    Nominal goal‑seeking PID controller for the DrivingContinuousRandom‑v0 car.

    Control     : u = [ ω  (rad/s clockwise +),
                        a  (m/s² forward    +) ]

    Call  `u = controller(state, dt)`  each step.
    """
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        kp_lin=0.002,  kd_lin=0.01,  ki_lin=0.02,
        kp_ang=1.0,  kd_ang=0.0,  ki_ang=0.0,
        a_bounds=(-0.25,  0.25),
        w_bounds=(-np.pi/4,  np.pi/4),
        accel_cutoff=np.deg2rad(90),        # no forward accel if |heading err|>90°
        int_clip=10.0                       # anti‑wind‑up limit
    ):
        self.kp_lin, self.kd_lin, self.ki_lin = kp_lin, kd_lin, ki_lin
        self.kp_ang, self.kd_ang, self.ki_ang = kp_ang, kd_ang, ki_ang

        self.a_min, self.a_max = a_bounds
        self.w_min, self.w_max = w_bounds

        self.cutoff      = accel_cutoff
        self.int_clip    = int_clip
        self.i_lin       = 0.0
        self.i_ang       = 0.0
        self.stuck_steps = 0                # for simple wall‑recovery

    # --------------------------------------------------------------------- #
    def __call__(self, s, dt=0.1):
        """
        Args
        ----
        s  : np.ndarray shape (11,) vehicle state
              [x, y, θ(clockwise+), vx, vy, ω, gx, gy, reached, crashed, …]
        dt : timestep [s]

        Returns
        -------
        u : np.ndarray(2,)  → [ω, a]
        """
        x, y, theta_cw = s[0:3]
        vx, vy         = s[3:5]
        omega_cw       = s[5]
        gx, gy         = s[6:8]

        # ---------- geometry ------------------------------------------------
        dx, dy    = gx - x, gy - y
        dist      = np.hypot(dx, dy)

        # goal direction in *mathematical* (CCW+) frame
        goal_dir_ccw = np.arctan2(dy, dx)
        # convert to CW‑positive angle
        goal_dir_cw  = goal_dir_ccw
        # goal_dir_cw = -np.pi/4
        # heading error in same (CW) sign convention, wrap to [-π, π]
        heading_err  = self._wrap(goal_dir_cw - theta_cw)

        # forward velocity (positive if moving along +heading)
        v_forward =  vx * np.cos(theta_cw) - vy * np.sin(theta_cw)

        # ---------- integral terms (anti‑wind‑up) ---------------------------
        self.i_lin = np.clip(self.i_lin + dist        * dt, -self.int_clip, self.int_clip)
        self.i_ang = np.clip(self.i_ang + heading_err * dt, -self.int_clip, self.int_clip)

        # ---------- PID -----------------------------------------------------
        # angular speed (ω, clockwise +)
        # w = ( self.kp_ang * heading_err
        #     - self.kd_ang * omega_cw
        #     + self.ki_ang * self.i_ang )
        w =  self.kp_ang * heading_err
        # linear acceleration (a, forward +)
        a = ( self.kp_lin * dist
            - self.kd_lin * v_forward )


        # ---------- heading‑aware throttle clamp ----------------------------
        # if abs(heading_err) > self.cutoff:        # > 90° off → no acceleration
        #     a = 0.0
        # else:                                     # smooth scaling (cosine)
        #     a *= np.cos(heading_err)

        # ---------- simple stuck recovery -----------------------------------
        # if abs(v_forward) < 0.05:
        #     self.stuck_steps += 1
        # else:
        #     self.stuck_steps  = 0
        #
        # if self.stuck_steps > 15:                 # ~1.5 s @ 10 Hz
        #     a = -1.0                               # back up
        #     w = np.sign(heading_err) * 4.0         # spin toward goal

        # ---------- saturate & return ---------------------------------------
        w = np.clip(w, self.w_min, self.w_max)
        a = np.clip(a, self.a_min, self.a_max)
        print(f"Current Heading is{theta_cw}, current heading error is {heading_err}")
        print(f"Current Goal heading is{goal_dir_ccw}, Goal heading actual is {goal_dir_cw}")
        print(f"dx={dx}, dy={dy}")
        print(f"w is {w}")
        return np.array([w, a])

    # --------------------------------------------------------------------- #
    @staticmethod
    def _wrap(angle):
        """wrap to [-π, π] with clockwise‑positive sign"""
        return (angle + np.pi) % (2*np.pi) - np.pi
# ──── QP‑filter class ────────────────────────────────────────────────────────────
class CBF_QP_Filter:
    """
    Quadratic‑program controller  (min ½‖u-u_nom‖²  s.t.  CBF + box bounds).
    * Assumes control‑affine system  ẋ = f(x) + g(x) u,  and g(x)=I on (x,y) dims.
    * Works with any torch nn.Module producing scalar h(x).
    """
    from collect_data import DubinsCar
    def __init__(self,
                 cbf_model: torch.nn.Module,
                 control_dim=2,
                 u_lower=(-np.pi/4., -0.25),
                 u_upper=( np.pi/4.,  0.25),
                 alpha=10.):
        self.cbf   = cbf_model.eval()
        self.udim  = control_dim
        self.u_lo  = torch.tensor(u_lower)
        self.u_hi  = torch.tensor(u_upper)
        self.alpha = alpha
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.norm_controller = PIDGoalController()
        self.dyn_model = DubinsCar()

    @torch.no_grad()
    def safe_action(self, states:np.ndarray, nn_input:np.ndarray):
        """
        state:  shape (state_dim,) ‑‑ raw observation of *one* agent.
        returns: numpy array (control_dim,)
        """
        with torch.enable_grad():
            x = torch.tensor(nn_input, dtype=torch.float32, requires_grad=True).to(self.device)

            # --- CBF value and gradient  (h,  ∂h/∂x) ---------------------------------
            h = self.cbf(x.unsqueeze(0)).squeeze()           # scalar

            dh_dx, = torch.autograd.grad(h, x, retain_graph=False)

        # Derivatives for affine system  (f ≈ 0, g = I on (x,y)):
        L_f_h = 0.0                                      # we ignore / set to 0
        L_g_h = dh_dx[0:self.udim]                       # take grad wrt x,y

        # Move everything to NumPy for cvxpy
        u_nom = self.norm_controller(states)
        A_cbf = -L_g_h.cpu().numpy().reshape(1, -1)      # A u ≤ b
        b_cbf =  (L_f_h + self.alpha * h).cpu().numpy()

        # --- build small dense QP ------------------------------------------------
        u   = cp.Variable(self.udim)
        Q   = np.eye(self.udim)
        obj = 0.5*cp.quad_form(u - u_nom, Q)
        constraints = [
            # A_cbf @ u <= b_cbf,               # CBF
            u >= self.u_lo.numpy(),           # box bounds
            u <= self.u_hi.numpy(),
        ]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(warm_start=True)

        # Fallback if the problem is infeasible / solver failed
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return u_nom.astype(np.float32)

        return u.value.astype(np.float32)
        # return u_nom.astype(np.float32)

sensor_model = ExponentialSensorModel(beta=1.0)
env = posggym.make(
    "DrivingContinuousRandom-v0",
    render_mode="human",
    obstacle_density=0.0,  # Increase density for more obstacles
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
if train:
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
        U_bounds=([-1, -1], [1, 1]),
        total_epoch=10
    )
    model, train_loss, test_loss = trainer.train()
else:
    model = CBFModel(env.observation_spaces["0"].shape[0]+3).to(device)
    state_dict = torch.load(load_dir, map_location=device)  # "cuda:0" if you like
    model.load_state_dict(state_dict)
    model.eval()

cbf_filter = CBF_QP_Filter(model, control_dim=2)

obs, _ = env.reset()
for t in range(1000):
    nn_ipt = np.concatenate([obs["0"],env.state[0].body[:3]])
    states = np.zeros((8,))
    states[:6] = env.state[0].body
    states[6:8] = env.state[0].dest_coord
    # nn_ipt = torch.from_numpy(nn_ipt).to(device).float()
    u_safe = cbf_filter.safe_action(states,nn_ipt)
    print(f"u_safe: {u_safe} is")
    actions = {"0": u_safe}
    step_result = env.step(actions)
    obs, rewards, terminations, truncations, _, infos = step_result
    # state = env.state[0].body[:3]
    env.render()
    if terminations["0"] or truncations["0"]:
        break
    time.sleep(0.1)
env.close()
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
# env.close()