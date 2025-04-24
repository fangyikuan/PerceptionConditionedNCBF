import torch
import math
from typing import Union
from torch.utils.data import Dataset
import numpy as np
class TrajectoryDataset(Dataset):
    """
    Normalises:
      • Observations   (dim = 2*n_sensors + 5)
      • Observation diff  (same rule)
      • States         (x,y,theta,vx,vy)

    and stores helper functions to apply / invert the same transforms later.
    """

    # ---------- constructor ----------
    def __init__(self,
                 data_dict,
                 n_sensors: int = 32,
                 obs_dist:   float = 5.0,
                 world_size: float = 14.0,
                 device: Union[str, torch.device] = "cpu"):

        self.device = torch.device(device)

        # 1) build OBSERVATION normaliser
        obs_dim = 2*n_sensors + 5             # 69 for default
        self.obs_scale  = torch.ones(obs_dim,  device=self.device)
        self.obs_offset = torch.zeros(obs_dim, device=self.device)

        # distances (wall + other vehicles) 0…d
        self.obs_scale[:2*n_sensors] = 1.0 / obs_dist

        # heading angle [-2π,2π]
        heading_idx = 2*n_sensors
        self.obs_scale[heading_idx] = 1.0 / (2*math.pi)

        # destination distances 0…s
        self.obs_scale[heading_idx+3: heading_idx+5] = 1.0 / world_size

        # 2) build STATE normaliser   [x, y, theta, vx, vy]
        self.state_scale  = torch.tensor(
            [1.0/world_size,     # x
             1.0/world_size,     # y
             1.0/(2*math.pi),    # theta
             1.0,                # vx  (-1..1 → unchanged)
             1.0],               # vy
            device=self.device
        )
        self.state_offset = torch.zeros_like(self.state_scale)

        # 3) convert & store tensors
        obs_raw      = torch.tensor(data_dict["observation"],      dtype=torch.float32, device=self.device)
        obs_diff_raw = torch.tensor(data_dict["observation_diff"], dtype=torch.float32, device=self.device)
        state_raw    = torch.tensor(data_dict["states"],           dtype=torch.float32, device=self.device)

        self.observations      = self._norm_obs(obs_raw)
        self.obs_diffs         = self._norm_obs(obs_diff_raw)
        self.states            = self._norm_state(state_raw)

        self.actions = torch.tensor(data_dict["action"], dtype=torch.float32, device=self.device)
        self.labels  = torch.tensor(data_dict["label"],  dtype=torch.float32, device=self.device)

    # ---------- normalisation helpers ----------
    def _norm_obs(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.obs_offset) * self.obs_scale

    def _denorm_obs(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.obs_scale - self.obs_offset

    def _norm_state(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.state_offset) * self.state_scale

    def _denorm_state(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.state_scale - self.state_offset

    # ---------- public API for rollout ----------
    def normalize_obs(self, raw_obs: torch.Tensor) -> torch.Tensor:
        return self._norm_obs(raw_obs.to(self.device))

    def denormalize_obs(self, norm_obs: torch.Tensor) -> torch.Tensor:
        return self._denorm_obs(norm_obs.to(self.device))

    def normalize_state(self, raw_state: torch.Tensor) -> torch.Tensor:
        return self._norm_state(raw_state.to(self.device))

    def denormalize_state(self, norm_state: torch.Tensor) -> torch.Tensor:
        return self._denorm_state(norm_state.to(self.device))

    # ---------- Dataset protocol ----------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "observation"      : self.observations[idx],
            "observation_diff" : self.obs_diffs[idx],
            "action"           : self.actions[idx],
            "state"            : self.states[idx],
            "label"            : self.labels[idx]
        }

class InputNormaliser:
    def __init__(self):

        # ------------------------------------------------------------------
        # CONSTANTS – change here if you alter sensor count or env geometry
        # ------------------------------------------------------------------
        self.N_SENSORS = 32
        self.OBS_DIST = 5.0  # d
        self.WORLD_SIZE = 14.0  # s

        # ------------------------------------------------------------------
        # build scale / offset vectors once
        # ------------------------------------------------------------------
        self.OBS_DIM = 2 * self.N_SENSORS + 5  # 69
        self.STATE_DIM = 5  # [x,y,theta,vx,vy]
        self.NN_DIM = self.OBS_DIM + self.STATE_DIM  # 74

        # observation scale / offset
        self.obs_scale = torch.ones(self.OBS_DIM)
        self.obs_offset = torch.zeros(self.OBS_DIM)

        self.obs_scale[:2 * self.N_SENSORS] = 1.0 / self.OBS_DIST  # distances
        self.heading_idx = 2 * self.N_SENSORS
        self.obs_scale[self.heading_idx] = 1.0 / (2 * math.pi)  # angle
        self.obs_scale[self.heading_idx + 3: self.heading_idx + 5] = 1.0 / self.WORLD_SIZE  # dest distances

        # state scale / offset
        state_scale = torch.tensor([
            1.0 / self.WORLD_SIZE,  # x
            1.0 / self.WORLD_SIZE,  # y
            1.0 / (2 * math.pi),  # theta
            1.0,  # vx   [-1,1] already
            1.0  # vy
        ])
        state_offset = torch.zeros_like(state_scale)

        # concatenated scale for NN input
        self.nn_scale = torch.cat([self.obs_scale, state_scale], dim=0)
        self.nn_offset = torch.cat([self.obs_offset, state_offset], dim=0)

        # ------------------------------------------------------------------
        # helper functions
        # ------------------------------------------------------------------

    def _to_tensor(self, x, device):
        """Convert ndarray → tensor, else leave as tensor; no copy if not needed."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device)

    def _to_numpy(self,x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert Tensor to CPU NumPy array; leave ndarray unchanged."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x  # already ndarray

    def normalize_obs(self,obs_raw: Union[torch.Tensor, np.ndarray],
                      device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """
        Normalise a raw observation (or batch). Shape (..., 69).
        Distances -> [0,1], angle -> [-1,1], vx/vy unchanged.
        """
        obs_raw = self._to_tensor(obs_raw, device)
        sc = self.obs_scale.to(device)
        of = self.obs_offset.to(device)
        return (obs_raw.to(device) + of) * sc

    def normalize_nn_input(self, nn_ipt: Union[torch.Tensor, np.ndarray],
                           device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """
        Normalise concatenated [obs_raw, state_raw]. Shape (..., 74).
        Uses the same rule as the dataset.
        """
        nn_ipt = self._to_tensor(nn_ipt, device)
        sc = self.nn_scale.to(device)
        of = self.nn_offset.to(device)
        return (nn_ipt.to(device) + of) * sc
