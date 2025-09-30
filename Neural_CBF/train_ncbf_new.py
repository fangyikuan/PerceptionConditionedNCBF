import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import os
from scipy.integrate import solve_ivp
from unicycle import DubinsCar


class CBFModel(nn.Module):
    def __init__(self, obs_dim):
        super(CBFModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

class NCBFTrainer:
    def __init__(self, obs_dim, state_dim, control_dim, training_data, test_data, U_bounds,
                 batchsize=32, total_epoch=20, lambda_param=1.0, mu=10.0, alpha=0, weight_decay=1e-6,
                 learning_rate = 0.003, bound_eps=0.2,
                 use_pgd=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        self.obs_dim = obs_dim # Obs_dim means the dim of NN input
        self.state_dim = state_dim # State dim means the dim of "state" space in control problem
        self.control_dim = control_dim
        self.batchsize = batchsize
        self.total_epoch = total_epoch
        self.lambda_param = lambda_param
        self.mu = mu
        self.alpha = alpha
        self.use_pgd = use_pgd
        self.eps = 1e-3
        self.bound_eps = bound_eps
        self.u_low = torch.tensor(U_bounds[0]).to(self.device)
        self.u_high = torch.tensor(U_bounds[1]).to(self.device)
        base_model = CBFModel(self.obs_dim+self.state_dim).to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs → DataParallel")
            self.model = torch.nn.DataParallel(base_model)
            # print("Model devices:", self.model.device_ids)
            # print("Parameter example on:", next(self.model.parameters()).device)
        else:
            self.model = base_model
        # self.model = CBFModel(self.obs_dim+self.state_dim).to(self.device)
        self.dyn_model = DubinsCar()

        self.optimizer = optim.NAdam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.2)

        self.train_loader = training_data
        self.test_loader = test_data

    def prepare_data(self, raw_data):
        x_data = torch.tensor(np.column_stack([d[0] for d in raw_data]), dtype=torch.float32).to(self.device)
        u_data = torch.tensor(np.column_stack([d[1] for d in raw_data]), dtype=torch.float32).to(self.device)
        y_data = torch.tensor(np.array([d[2] for d in raw_data]), dtype=torch.float32).to(self.device)
        return TensorDataset(x_data.t(), u_data.t(), y_data.reshape(1, -1).t())

    def relu(self, x):
        return torch.maximum(x, torch.zeros_like(x))

    def sigmoid_fast(self, x):
        return torch.sigmoid(x)

    def f_batch(self, A, x):
        return torch.matmul(A, x.unsqueeze(-1)).squeeze(-1) if A.dim() == 2 else torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

    def g_batch(self, B, u):
        return torch.matmul(B, u.unsqueeze(-1)).squeeze(-1) if B.dim() == 2 else torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)

    def affine_dyn_batch(self, A, x, B, u, Delta=None):
        x_dot = self.f_batch(A, x) + self.g_batch(B, u)
        if Delta is not None:
            x_dot += Delta
        return x_dot

    def forward_invariance_func(self,  obs_b, obs_diff_b, state_b, action_b, A, B, Delta=None):
        # x_grad = obs_b.clone().detach().requires_grad_(True)
        x_grad = torch.cat([obs_b, state_b], -1).detach().requires_grad_(True)
        phi_x = self.model(x_grad)
        gradients = torch.autograd.grad(phi_x, x_grad, grad_outputs=torch.ones_like(phi_x), create_graph=True)[0]
        x_dot = self.affine_dyn_batch(A, state_b, B, action_b, Delta)
        x_dot = torch.cat([obs_diff_b, x_dot], dim=-1)
        phi_dot = torch.sum(gradients * x_dot, dim=1, keepdim=True)
        return phi_dot + self.alpha * self.model(torch.cat([obs_b, state_b], dim=-1))

    def forward_invariance_func_noAB(self, obs_b, obs_diff_b, state_b, action_b, alpha=0):
        """
        Compute the time derivative of phi along given trajectories plus a decay term: ϕ̇ + α*ϕ
        """
        batch_size = obs_b.shape[0]
        state_dim = obs_b.shape[1]

        # We need a clone of x with requires_grad=True for computing gradients
        x_grad = obs_b.clone().detach().requires_grad_(True)
        x_grad = torch.cat((x_grad, state_b), dim=-1)
        phi_x = self.model(x_grad)

        # Compute gradient of phi with respect to x
        ones = torch.ones_like(phi_x, device=obs_b.device)
        gradients = torch.autograd.grad(
            outputs=phi_x,
            inputs=x_grad,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        x_dot = self.dyn_model.dynamics(state_b, action_b, npy=False)
        x_dot = torch.cat([obs_diff_b, x_dot], dim=-1)
        # Compute ϕ̇ = ∇ϕᵀẋ (dot product of gradient and dynamics)
        phi_dot = torch.sum(gradients * x_dot, dim=1, keepdim=True)

        # Compute ϕ̇ + α*ϕ
        l = phi_dot + self.alpha * self.model(torch.cat([obs_b, state_b], dim=-1))

        return l

    def loss_naive_safeset(self, x, y_init):
        # y_init = y_init.squeeze(1)
        phi_x = self.model(x).squeeze(1)
        return torch.mean(self.relu((2 * y_init -1) * phi_x + 0.01))

    def loss_regularization(self, x, y_init):
        # y_init = y_init.squeeze(1)
        phi_x = self.model(x).squeeze(1)
        return torch.mean(self.sigmoid_fast((2 * y_init -1 ) * phi_x))

    def loss_naive_fi(self, obs_b, obs_diff_b, state_b, action_b, A, B, y_init, Delta=None, epsilon=1):
        # y_init = y_init.squeeze(1)
        safe_indices = (y_init == 1).nonzero(as_tuple=True)[0]
        if len(safe_indices) == 0: return torch.tensor(0.0, device=self.device)

        obs_safe, obs_diff_safe, x_safe, u_safe= (obs_b[safe_indices], obs_diff_b[safe_indices], state_b[safe_indices],
                                                  action_b[safe_indices])
        A_safe, B_safe = A[safe_indices], B[safe_indices]
        Delta_safe = Delta[safe_indices] if Delta is not None else None

        with torch.no_grad():
            phi_x = self.model(torch.cat([obs_safe, x_safe],dim=-1))

        boundary_indices = (torch.abs(phi_x) < epsilon).squeeze(1).nonzero(as_tuple=True)[0]
        if len(boundary_indices) == 0: return torch.tensor(0.0, device=self.device)
        obs_bd, obs_diff_bd,x_bd, u_bd = (obs_b[boundary_indices], obs_diff_b[boundary_indices], state_b[boundary_indices],
                                                  action_b[boundary_indices])
        # x_b, u_b = x_safe[boundary_indices], u_safe[boundary_indices]
        A_b, B_b = A_safe[boundary_indices], B_safe[boundary_indices]
        Delta_b = Delta_safe[boundary_indices] if Delta_safe is not None else None

        if self.use_pgd:
            u_b = self.pgd_find_u_notce(x_bd, A_b, B_b, u_bd, Delta_b)

        fi_vals = self.forward_invariance_func_noAB(obs_bd, obs_diff_bd, x_bd, u_bd)
        return torch.mean(self.relu(fi_vals + 1e-6))

    def pgd_find_u_notce(self, x, A, B, u_0, Delta=None, lr=1, num_iter=10):
        u = u_0.clone().detach().requires_grad_(True)
        optimizer = optim.SGD([u], lr=lr)
        low_U, high_U = self.u_low.to(self.device), self.u_high.to(self.device)

        for _ in range(num_iter):
            optimizer.zero_grad()
            fi_vals = self.forward_invariance_func(x, A, B, u, Delta)
            loss = torch.mean(fi_vals)
            loss.backward(retain_graph=True)
            optimizer.step()
            with torch.no_grad():
                u.data = torch.clamp(u.data, min=low_U.expand_as(u), max=high_U.expand_as(u))
        return u.detach()

    def compute_dynamics_and_residuals(self, x_batch, u_batch):
        A_batch, B_batch = self.dyn_model.jacobian(x_batch, u_batch)
        Delta = torch.zeros(x_batch.shape[0], self.state_dim, device=self.device)
        x_eps = x_batch - self.eps  # or better: use torch.randn_like(...)
        u_eps = u_batch  # keep the same control
        f_actual = self.dyn_model.dynamics(x_eps, u_eps, npy=False)
        f_linear = torch.einsum("bij,bj->bi", A_batch, x_eps) + \
                   torch.einsum("bij,bj->bi", B_batch, u_eps)
        Delta = f_actual - f_linear  # shape (B, 5)
        return A_batch, B_batch, Delta

    def train(self):
        train_losses, test_losses = [], []

        for epoch in range(1, self.total_epoch + 1):
            self.model.train()
            train_epoch_loss = []

            phi_safe_epoch = []
            phi_unsafe_epoch = []

            for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):
                obs_batch = batch["observation"].to(self.device)
                obs_diff_batch = batch["observation_diff"].to(self.device)
                action_batch = batch["action"].to(self.device)
                state_batch = batch["state"].to(self.device)
                label_batch = batch["label"].to(self.device)
                nn_input_batch = torch.cat([obs_batch, state_batch],dim=-1)
                A, B, Delta = self.compute_dynamics_and_residuals(state_batch, action_batch)
                if self.use_pgd:
                    action_batch = self.pgd_find_u_notce(state_batch, A, B, action_batch, Delta)
                self.optimizer.zero_grad()
                safe_loss = self.loss_naive_safeset(nn_input_batch, label_batch)
                fi_loss = self.loss_naive_fi(obs_batch, obs_diff_batch, state_batch, action_batch, A,
                                                              B, label_batch, Delta, epsilon=self.bound_eps )
                reg_loss = self.loss_regularization(nn_input_batch, label_batch)
                loss = safe_loss + \
                       self.lambda_param * fi_loss+ \
                       self.mu * reg_loss
                # print(f"current safe loss is {safe_loss}, fi loss is {fi_loss}, reg loss is {reg_loss}")
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())
                with torch.no_grad():
                    phi_x = self.model(nn_input_batch)

                safe_mask = (label_batch == 1)
                unsafe_mask = ~safe_mask

                if safe_mask.any():
                    phi_safe_epoch.append(phi_x[safe_mask])
                if unsafe_mask.any():
                    phi_unsafe_epoch.append(phi_x[unsafe_mask])

            print(f"current safe loss is {safe_loss}, fi loss is {fi_loss}, reg loss is {reg_loss}")
            self.scheduler.step()
            avg_train = np.mean(train_epoch_loss)
            train_losses.append(avg_train)
            if phi_safe_epoch and phi_unsafe_epoch:
                phi_safe_epoch = torch.cat(phi_safe_epoch)
                phi_unsafe_epoch = torch.cat(phi_unsafe_epoch)

                phi_min = min(phi_safe_epoch.min(), phi_unsafe_epoch.min()).item()
                phi_max = max(phi_safe_epoch.max(), phi_unsafe_epoch.max()).item()
                frac_bd = (torch.cat([phi_safe_epoch, phi_unsafe_epoch]).abs() < self.bound_eps).float().mean().item()

                print(f"[epoch {epoch:02d}] ϕ range {phi_min:+.3f} … {phi_max:+.3f}  "
                      f"|ϕ|<ε({self.bound_eps})={frac_bd:.2%}  "
                      f"μ_safe={phi_safe_epoch.mean().item():+.3f}  "
                      f"μ_unsafe={phi_unsafe_epoch.mean().item():+.3f}")

            self.model.eval()
            test_epoch_loss = []
            for batch in tqdm(self.test_loader, desc=f"Test Epoch {epoch}"):
                obs_batch = batch["observation"].to(self.device)
                obs_diff_batch = batch["observation_diff"].to(self.device)
                action_batch = batch["action"].to(self.device)
                state_batch = batch["state"].to(self.device)
                label_batch = batch["label"].to(self.device)
                nn_input_batch = torch.cat([obs_batch, state_batch], dim=-1)
                A, B, Delta = self.compute_dynamics_and_residuals(state_batch, action_batch)
                loss = self.loss_naive_safeset(nn_input_batch, label_batch) + \
                       self.lambda_param * self.loss_naive_fi(obs_batch, obs_diff_batch, state_batch, action_batch, A,
                                                              B, label_batch, Delta) + \
                       self.mu * self.loss_regularization(nn_input_batch, label_batch)
                test_epoch_loss.append(loss.item())
            avg_test = np.mean(test_epoch_loss)
            test_losses.append(avg_test)
            # -- after every epoch --

            print(f"Epoch {epoch}: Train Loss = {avg_train:.4f}, Test Loss = {avg_test:.4f}")
            # torch.save(self.model.state_dict(), f"cbf_model_epoch_{epoch}.pt")
            if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                torch.save(self.model.module.state_dict(), f"./cbf_model_epoch_{epoch}.pt")
            else:
                torch.save(self.model.state_dict(), f"./cbf_model_epoch_{epoch}.pt")

        self.plot_loss(train_losses, test_losses)
        return self.model, train_losses, test_losses

    def plot_loss(self, train_loss, test_loss):
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Train")
        plt.plot(test_loss, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("NCBF Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.savefig("ncbf_loss_curves.png")
        plt.show()
