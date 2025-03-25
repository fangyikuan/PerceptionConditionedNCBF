import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from collect_data import DubinsCar, Hyperrectangle
import os

class CBFModel(nn.Module):
    def __init__(self, state_dim):
        super(CBFModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

class NCBFTrainer:
    def __init__(self, state_dim, control_dim, training_data, test_data, U_bounds,
                 batchsize=128, total_epoch=20, lambda_param=1.0, mu=0.1, alpha=0.0, use_pgd=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.batchsize = batchsize
        self.total_epoch = total_epoch
        self.lambda_param = lambda_param
        self.mu = mu
        self.alpha = alpha
        self.use_pgd = use_pgd
        self.eps = 1e-3
        self.U = Hyperrectangle(U_bounds[0], U_bounds[1], npy=False)

        self.model = CBFModel(state_dim).to(self.device)
        self.dyn_model = DubinsCar()

        self.optimizer = optim.NAdam(self.model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.2)

        self.train_loader = DataLoader(self.prepare_data(training_data), batch_size=batchsize, shuffle=True)
        self.test_loader = DataLoader(self.prepare_data(test_data), batch_size=batchsize, shuffle=True)

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

    def forward_invariance_func(self, x, A, B, u, Delta=None):
        x_grad = x.clone().detach().requires_grad_(True)
        phi_x = self.model(x_grad)
        gradients = torch.autograd.grad(phi_x, x_grad, grad_outputs=torch.ones_like(phi_x), create_graph=True)[0]
        x_dot = self.affine_dyn_batch(A, x, B, u, Delta)
        phi_dot = torch.sum(gradients * x_dot, dim=1, keepdim=True)
        return phi_dot + self.alpha * self.model(x)

    def loss_naive_safeset(self, x, y_init):
        y_init = y_init.squeeze(1)
        phi_x = self.model(x).squeeze(1)
        return torch.mean(self.relu((2 * y_init - 1) * phi_x + 1e-6))

    def loss_regularization(self, x, y_init):
        y_init = y_init.squeeze(1)
        phi_x = self.model(x).squeeze(1)
        return torch.mean(self.sigmoid_fast((2 * y_init - 1) * phi_x))

    def loss_naive_fi(self, x, A, B, u, y_init, Delta=None, epsilon=0.1):
        y_init = y_init.squeeze(1)
        safe_indices = (y_init == 1).nonzero(as_tuple=True)[0]
        if len(safe_indices) == 0: return torch.tensor(0.0, device=self.device)

        x_safe, u_safe = x[safe_indices], u[safe_indices]
        A_safe, B_safe = A[safe_indices], B[safe_indices]
        Delta_safe = Delta[safe_indices] if Delta is not None else None

        with torch.no_grad():
            phi_x = self.model(x_safe)

        boundary_indices = (torch.abs(phi_x) < epsilon).squeeze(1).nonzero(as_tuple=True)[0]
        if len(boundary_indices) == 0: return torch.tensor(0.0, device=self.device)

        x_b, u_b = x_safe[boundary_indices], u_safe[boundary_indices]
        A_b, B_b = A_safe[boundary_indices], B_safe[boundary_indices]
        Delta_b = Delta_safe[boundary_indices] if Delta_safe is not None else None

        if self.use_pgd:
            u_b = self.pgd_find_u_notce(x_b, A_b, B_b, u_b, Delta_b)

        fi_vals = self.forward_invariance_func(x_b, A_b, B_b, u_b, Delta=Delta_b)
        return torch.mean(self.relu(fi_vals + 1e-6))

    def pgd_find_u_notce(self, x, A, B, u_0, Delta=None, lr=1, num_iter=10):
        u = u_0.clone().detach().requires_grad_(True)
        optimizer = optim.SGD([u], lr=lr)
        low_U, high_U = self.U.low.to(self.device), self.U.high.to(self.device)

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
        for i in range(x_batch.shape[0]):
            x_eps = x_batch[i] - self.eps
            u_eps = u_batch[i] - self.eps
            f_actual = self.dyn_model.dynamics(x_eps, u_eps, npy=False)
            f_linear = A_batch[i] @ x_eps + B_batch[i] @ u_eps
            Delta[i] = f_actual - f_linear
        return A_batch, B_batch, Delta

    def train(self):
        train_losses, test_losses = [], []

        for epoch in range(1, self.total_epoch + 1):
            self.model.train()
            train_epoch_loss = []
            for x, u, y in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):
                A, B, Delta = self.compute_dynamics_and_residuals(x, u)
                if self.use_pgd:
                    u = self.pgd_find_u_notce(x, A, B, u, Delta)
                self.optimizer.zero_grad()
                loss = self.loss_naive_safeset(x, y) + \
                       self.lambda_param * self.loss_naive_fi(x, A, B, u, y, Delta) + \
                       self.mu * self.loss_regularization(x, y)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())
            self.scheduler.step()
            avg_train = np.mean(train_epoch_loss)
            train_losses.append(avg_train)

            self.model.eval()
            test_epoch_loss = []
            for x, u, y in tqdm(self.test_loader, desc=f"Test Epoch {epoch}"):
                A, B, Delta = self.compute_dynamics_and_residuals(x, u)
                loss = self.loss_naive_safeset(x, y) + \
                       self.lambda_param * self.loss_naive_fi(x, A, B, u, y, Delta) + \
                       self.mu * self.loss_regularization(x, y)
                test_epoch_loss.append(loss.item())
            avg_test = np.mean(test_epoch_loss)
            test_losses.append(avg_test)

            print(f"Epoch {epoch}: Train Loss = {avg_train:.4f}, Test Loss = {avg_test:.4f}")
            torch.save(self.model.state_dict(), f"cbf_model_epoch_{epoch}.pt")

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
