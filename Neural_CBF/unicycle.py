import numpy as np
import torch

import cvxpy as cp

class DubinsCar:
    """
    State:  [x, y, theta, vx, vy]
    Control: [a (longitudinal accel), omega (yaw rate)]
    Supports both single-state/vector and batched inputs.
    """

    def __init__(self):
        self.state_dim   = 5   # x, y, theta, vx, vy
        self.control_dim = 2   # a, omega
    def dynamics(self, x, u, npy=True):
        """
        Continuous-time dynamics:
            x_dot = [vx, vy, omega, a*cos(theta) - omega*vy, a*sin(theta) + omega*vx]

        Accepts:
          x: shape (..., state_dim)
          u: shape (..., control_dim)
          npy: if True, use numpy; else torch
        Returns:
          x_dot: same shape as x
        """
        if npy:
            x_arr = np.asarray(x)
            u_arr = np.asarray(u)
            # extract components
            theta = x_arr[..., 2]
            vx    = x_arr[..., 3]
            vy    = x_arr[..., 4]
            a     = u_arr[..., 0]
            omega = u_arr[..., 1]

            # compute derivatives
            vx_dot = a * np.cos(theta) - omega * vy
            vy_dot = a * np.sin(theta) + omega * vx

            # stack results
            x_dot = np.stack([vx,
                              vy,
                              omega,
                              vx_dot,
                              vy_dot], axis=-1)
        else:
            x_t = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
            u_t = u if torch.is_tensor(u) else torch.tensor(u, dtype=torch.float32)

            theta = x_t[..., 2]
            vx    = x_t[..., 3]
            vy    = x_t[..., 4]
            a     = u_t[..., 0]
            omega = u_t[..., 1]

            vx_dot = a * torch.cos(theta) - omega * vy
            vy_dot = a * torch.sin(theta) + omega * vx

            x_dot = torch.stack([vx,
                                 vy,
                                 omega,
                                 vx_dot,
                                 vy_dot], dim=-1)

        return x_dot

    def f(self, x, npy=True):
        """
        Drift term f(x): shape (..., 5)
        f(x) = [vx, vy, 0, 0, 0]
        """
        if npy:
            x_arr = np.asarray(x)
            vx = x_arr[..., 3]
            vy = x_arr[..., 4]
            z = np.zeros_like(vx)
            return np.stack([vx, vy, z, z, z], axis=-1)
        else:
            x_t = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
            vx = x_t[..., 3]
            vy = x_t[..., 4]
            z = torch.zeros_like(vx)
            return torch.stack([vx, vy, z, z, z], dim=-1)

    def g(self, x, npy=True):
        """
        Input matrix g(x): shape (..., 5, 2)
        Columns correspond to controls [a, omega]:
          g[:, :, 0] = [0, 0, 0, cos(theta), sin(theta)]^T
          g[:, :, 1] = [0, 0, 1, -vy,        vx       ]^T
        """
        if npy:
            x_arr = np.asarray(x)
            theta = x_arr[..., 2]
            vx = x_arr[..., 3]
            vy = x_arr[..., 4]

            g_a = np.stack([np.zeros_like(theta),
                            np.zeros_like(theta),
                            np.zeros_like(theta),
                            np.cos(theta),
                            np.sin(theta)], axis=-1)  # (..., 5)

            g_om = np.stack([np.zeros_like(theta),
                             np.zeros_like(theta),
                             np.ones_like(theta),
                             -vy,
                             vx], axis=-1)  # (..., 5)

            G = np.stack([g_a, g_om], axis=-1)  # (..., 5, 2)
            return G
        else:
            x_t = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
            theta = x_t[..., 2]
            vx = x_t[..., 3]
            vy = x_t[..., 4]

            zeros = torch.zeros_like(theta)
            ones = torch.ones_like(theta)

            g_a = torch.stack([zeros, zeros, zeros, torch.cos(theta), torch.sin(theta)], dim=-1)  # (..., 5)
            g_om = torch.stack([zeros, zeros, ones, -vy, vx], dim=-1)  # (..., 5)

            G = torch.stack([g_a, g_om], dim=-1)  # (..., 5, 2)
            return G

    def jacobian(self, x, u):
        """
        Compute A = df/dx and B = df/du for batched torch inputs.
        Input shapes: x (batch, 5), u (batch, 2)
        Returns: A (batch, 5, 5), B (batch, 5, 2)
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(u):
            u = torch.tensor(u, dtype=torch.float32)

        # ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)

        batch = x.shape[0]
        A = torch.zeros(batch, self.state_dim, self.state_dim, device=x.device)
        B = torch.zeros(batch, self.state_dim, self.control_dim, device=x.device)

        theta = x[:, 2]
        vx    = x[:, 3]
        vy    = x[:, 4]
        a     = u[:, 0]
        omega = u[:, 1]

        # f0 = x_dot = vx
        A[:, 0, 3] = 1.0
        # f1 = y_dot = vy
        A[:, 1, 4] = 1.0
        # f2 = theta_dot = omega -> no state partials
        # f3 = vx_dot = a*cos(theta) - omega*vy
        A[:, 3, 2] = -a * torch.sin(theta)   # ∂f3/∂theta
        A[:, 3, 4] = -omega                  # ∂f3/∂vy
        # f4 = vy_dot = a*sin(theta) + omega*vx
        A[:, 4, 2] = a * torch.cos(theta)    # ∂f4/∂theta
        A[:, 4, 3] = omega                   # ∂f4/∂vx

        # B: control jacobian
        B[:, 2, 1] = 1.0                     # ∂theta_dot/∂omega
        B[:, 3, 0] = torch.cos(theta)        # ∂vx_dot/∂a
        B[:, 3, 1] = -vy                     # ∂vx_dot/∂omega
        B[:, 4, 0] = torch.sin(theta)        # ∂vy_dot/∂a
        B[:, 4, 1] = vx                      # ∂vy_dot/∂omega

        return A, B

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
                 u_lower=(-0, -np.pi/4),
                 u_upper=(0.25,  np.pi/4),
                 alpha=10.,
                 slack=5):
        self.cbf   = cbf_model.eval()
        self.udim  = control_dim
        self.xdim  = 5
        self.u_lo  = torch.tensor(u_lower)
        self.u_hi  = torch.tensor(u_upper)
        self.alpha = alpha
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.norm_controller = PIDGoalController()
        self.dyn_model = DubinsCar()
        self.cbf_slack_w = slack

    @torch.no_grad()
    def safe_action(self,
                    state: np.ndarray,
                    nn_input: torch.Tensor,
                    diff_perception: np.ndarray) -> np.ndarray:
        """
        state : raw vehicle state (len = self.xdim + extras)
        returns a numpy array [ω , a]
        """
        # 1) CBF value + gradient -------------------------------------------------
        with torch.enable_grad():
            # x = torch.tensor(nn_input, dtype=torch.float32,
            #                  device=self.device, requires_grad=True)
            x = nn_input.requires_grad_()
            h = self.cbf(x.unsqueeze(0)).squeeze()
            dh_dx, = torch.autograd.grad(h, x)

        # split ∇h into perception-part / physical-state part
        dh_p = dh_dx[:-self.xdim].cpu().numpy().reshape(1, -1)  # (1,P)
        dh_x = dh_dx[-self.xdim:].cpu().numpy().reshape(1, -1)  # (1,xdim)

        # 2) nominal control ------------------------------------------------------
        u_nom_aw = self.norm_controller(state)  # [ω , a]  (AW = actuator-world)
        u_nom_qp = u_nom_aw[[1, 0]].copy()  # [a , ω]  (QP order)

        # 3) system Jacobians -----------------------------------------------------
        x_t = state[:self.xdim]
        u_t = u_nom_qp.copy()
        # x_t = torch.as_tensor(state[:self.xdim], dtype=torch.float32)
        # u_t = torch.as_tensor(u_nom_qp, dtype=torch.float32)
        A, B = self.dyn_model.jacobian(x_t, u_t)  # torch, shapes (xdim,xdim) & (xdim,udim)
        A, B = A.cpu().numpy().squeeze(0), B.cpu().numpy().squeeze(0)

        f_t = self.dyn_model.f(x_t)  # drift term
        g_t = self.dyn_model.g(x_t)  # input matrix
        drift_t = f_t + g_t @ u_t  # true dot{x_t}
        b = drift_t - (A @ x_t + B @ u_t)  # affine offset term
        # 4) CBF linear constraint  A_cbf u ≥ -B_cbf
        A_cbf = dh_x @ B  # (1,udim)
        Lf_h = (dh_x @ (A @ state[:self.xdim].reshape(-1, 1) + b.reshape(-1, 1))
                + dh_p @ diff_perception.reshape(-1, 1))

        # B_cbf = L_f h + α(h)
        B_cbf = Lf_h + self.alpha * h.item()

        # 5) QP -------------------------------------------------------------------
        u = cp.Variable(self.udim)  # [a, ω]
        Q = np.eye(self.udim)
        obj = 0.5 * cp.quad_form(u - u_nom_qp, Q)
        sigma = cp.Variable(nonneg=True)
        obj += self.cbf_slack_w * sigma

        constr = [A_cbf @ u + sigma >= -B_cbf,
                  u >= self.u_lo.cpu().numpy(),
                  u <= self.u_hi.cpu().numpy()]

        prob = cp.Problem(cp.Minimize(obj), constr)
        prob.solve(solver=cp.OSQP, warm_start=True)
        print(f"H:{h.item()}")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            # return nominal (converted back to [ω , a])
            return u_nom_aw.astype(np.float32)

        u_qp = u.value.astype(np.float32)  # [a, ω] from solver
        return u_qp[[1, 0]]  # convert → [ω , a]
