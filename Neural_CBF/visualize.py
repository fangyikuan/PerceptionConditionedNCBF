import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from train_ncbf import  pgd_find_u_notce,forward_invariance_func
from collect_data import Hyperrectangle, DubinsCar

def verification_forward(model, A, x, B, u_0, U, alpha=0, lr=1, num_iter=10, Delta=None):
    """Verify forward invariance by finding best-case control inputs"""
    # Compute forward invariance with initial control
    original_condition = forward_invariance_func(model, A, x, B, u_0, alpha, Delta) <= 0
    
    # Find best-case control
    u = pgd_find_u_notce(model, A, x, B, u_0, U, alpha, lr, num_iter, Delta)
    
    # Compute forward invariance with best-case control
    best_condition = forward_invariance_func(model, A, x, B, u, alpha, Delta) <= 0
    opt_condition = forward_invariance_func(model, A, x, B, u, alpha, Delta)
    
    return original_condition, best_condition, u, opt_condition

def Phi_naive_car(model, x, y, theta=0):
    """Compute CBF value at given state"""
    # Create input with shape [batch_size, state_dim]
    if np.isscalar(x) and np.isscalar(y):
        # Single point
        input_data = np.array([[x, y, theta]])
    else:
        # Create meshgrid for 2D visualization
        X, Y = np.meshgrid(x, y)
        input_data = np.stack([X.flatten(), Y.flatten(), np.ones_like(X.flatten()) * theta], axis=1)
    
    # Convert to torch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # Evaluate model
    with torch.no_grad():
        phi_values = model(input_tensor).numpy()
    
    # Reshape if necessary
    if not np.isscalar(x) and not np.isscalar(y):
        phi_values = phi_values.reshape(len(y), len(x))
    
    return phi_values

def Phi_dot_naive_car(model, dyn_model, x, y, alpha=0, theta=0, v=0, w=0):
    """Compute time derivative of CBF at given state"""
    eps = 1e-3
    
    # Create input with shape [batch_size, state_dim]
    if np.isscalar(x) and np.isscalar(y):
        # Single point
        input_data = np.array([[x, y, theta]])
        u_data = np.array([[v, w]])
    else:
        # Create meshgrid for 2D visualization
        X, Y = np.meshgrid(x, y)
        input_data = np.stack([X.flatten(), Y.flatten(), np.ones_like(X.flatten()) * theta], axis=1)
        u_data = np.stack([np.ones_like(X.flatten()) * v, np.ones_like(X.flatten()) * w], axis=1)
    
    # Number of points
    batch_size = input_data.shape[0]
    
    # Initialize arrays for A, B, and Delta
    A_batch = np.zeros((batch_size, dyn_model.state_dim, dyn_model.state_dim))
    B_batch = np.zeros((batch_size, dyn_model.state_dim, dyn_model.control_dim))
    Delta = np.zeros((batch_size, dyn_model.state_dim))
    
    # Compute linearization for each point
    for i in range(batch_size):
        A, B = dyn_model.jacobian(input_data[i], u_data[i])
        A_batch[i] = A
        B_batch[i] = B
        
        # Compute residual term (Δ)
        x_eps = input_data[i] - eps
        u_eps = u_data[i] - eps
        
        # Compute actual dynamics
        f_actual = dyn_model.dynamics(x_eps, u_eps, npy=False)
        
        # Compute linearized dynamics
        f_linear = np.matmul(A, x_eps) + np.matmul(B, u_eps)
        
        # Residual is the difference
        Delta[i] = f_actual - f_linear
    
    # Compute forward invariance condition
    A_batch = torch.tensor(A_batch,dtype=torch.float32)
    B_batch = torch.tensor(B_batch,dtype=torch.float32)
    phi_dot = forward_invariance_func(model, A_batch, input_data, B_batch, u_data, alpha=alpha, Delta=Delta)
    
    # Reshape if necessary
    if not np.isscalar(x) and not np.isscalar(y):
        phi_dot = phi_dot.reshape(len(y), len(x))
    
    return phi_dot

def h_naive_car(model, dyn_model, U, x, y, alpha=0, theta=0, v=0, w=0, lr=0.1, num_iter=10):
    """Compute best-case CBF derivative at given state"""
    eps = 1e-3
    
    # Create input with shape [batch_size, state_dim]
    if np.isscalar(x) and np.isscalar(y):
        # Single point
        input_data = np.array([[x, y, theta]])
        u_data = np.array([[v, w]])
    else:
        # Create meshgrid for 2D visualization
        X, Y = np.meshgrid(x, y)
        input_data = np.stack([X.flatten(), Y.flatten(), np.ones_like(X.flatten()) * theta], axis=1)
        u_data = np.stack([np.ones_like(X.flatten()) * v, np.ones_like(X.flatten()) * w], axis=1)
    
    # Number of points
    batch_size = input_data.shape[0]
    
    # Initialize arrays for A, B, and Delta
    A_batch = np.zeros((batch_size, dyn_model.state_dim, dyn_model.state_dim))
    B_batch = np.zeros((batch_size, dyn_model.state_dim, dyn_model.control_dim))
    Delta = np.zeros((batch_size, dyn_model.state_dim))
    
    # Compute linearization for each point
    for i in range(batch_size):
        A, B = dyn_model.jacobian(input_data[i], u_data[i])
        A_batch[i] = A
        B_batch[i] = B
        
        # Compute residual term (Δ)
        x_eps = input_data[i] - eps
        u_eps = u_data[i] - eps
        
        # Compute actual dynamics
        f_actual = dyn_model.dynamics(x_eps, u_eps, npy=False)
        
        # Compute linearized dynamics
        f_linear = np.matmul(A, x_eps) + np.matmul(B, u_eps)
        
        # Residual is the difference
        Delta[i] = f_actual - f_linear
    
    # Find best-case control input
    _, _, _, opt_condition = verification_forward(
        model, A_batch, input_data, B_batch, u_data, U, 
        alpha=alpha, lr=lr, num_iter=num_iter, Delta=Delta
    )
    
    # Reshape if necessary
    if not np.isscalar(x) and not np.isscalar(y):
        opt_condition = opt_condition.reshape(len(y), len(x))
    
    return opt_condition

def plot_env(X, X_unsafe):
    """Plot environment with safe and unsafe regions"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot safe region
    ax.add_patch(plt.Rectangle(
        (X.low[0], X.low[1]), 
        X.high[0] - X.low[0], 
        X.high[1] - X.low[1],
        fill=False, edgecolor='blue', linewidth=2
    ))
    
    # Plot unsafe region
    ax.add_patch(plt.Rectangle(
        (X_unsafe.low[0], X_unsafe.low[1]), 
        X_unsafe.high[0] - X_unsafe.low[0], 
        X_unsafe.high[1] - X_unsafe.low[1],
        fill=True, facecolor='red', alpha=0.3, edgecolor='red', linewidth=2
    ))
    
    ax.set_xlim(X.low[0], X.high[0])
    ax.set_ylim(X.low[1], X.high[1])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    return fig, ax

# Main visualization code
def visualize_cbf(model, alpha=0.0, theta=0, v=1, w=1, pgd_lr=10, pgd_num_iter=10):
    # Define safe and unsafe regions
    X = Hyperrectangle(low=[0, 0, 0], high=[4, 4, np.pi])
    X_unsafe = Hyperrectangle(low=[1.5, 0, 0], high=[2.5, 2, np.pi])
    U = Hyperrectangle(low=[-1, -1], high=[1, 1])
    
    # Create Dubins car model
    dyn_model = DubinsCar()
    
    # Create grid for visualization
    x = np.linspace(0, 4, 50)
    y = np.linspace(0, 4, 50)
    
    # Compute CBF derivative
    print("Computing CBF derivative...")
    z1 = np.zeros((len(y), len(x)))
    for i, yi in tqdm(enumerate(y)):
        for j, xi in enumerate(x):
            z1[i, j] = Phi_dot_naive_car(model, dyn_model, xi, yi, alpha=alpha, theta=theta, v=v, w=w)[0, 0]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot 1: Contour of CBF derivative
    _, ax1 = plot_env(X, X_unsafe)
    axes[0, 0] = ax1
    contour = ax1.contourf(np.meshgrid(x, y)[0], np.meshgrid(x, y)[1], z1, levels=10, cmap='turbo')
    ax1.contour(np.meshgrid(x, y)[0], np.meshgrid(x, y)[1], z1, levels=[0], colors='black', linewidths=2)
    ax1.set_title('CBF Derivative Contour')
    plt.colorbar(contour, ax=ax1)
    plt.savefig('CBF_Derivative_Contour.png', dpi=300)
    
    # Plot 2: Heatmap of CBF derivative
    _, ax2 = plot_env(X, X_unsafe)
    axes[0, 1] = ax2
    heatmap = ax2.imshow(z1, extent=[x.min(), x.max(), y.min(), y.max()], 
                        origin='lower', aspect='auto', cmap='turbo')
    ax2.set_title('CBF Derivative Heatmap')
    plt.colorbar(heatmap, ax=ax2)
    plt.savefig('CBF_Derivative_Heatmap.png', dpi=300)
    
    # Compute best-case CBF derivative
    print("Computing best-case CBF derivative...")
    z2 = np.zeros((len(y), len(x)))
    for i, yi in tqdm(enumerate(y)):
        for j, xi in enumerate(x):
            z2[i, j] = h_naive_car(
                model, dyn_model, U, xi, yi, 
                alpha=alpha, theta=theta, v=v, w=w, 
                lr=pgd_lr, num_iter=pgd_num_iter
            )[0, 0]
    
    # Plot 3: Contour of best-case CBF derivative
    _, ax3 = plot_env(X, X_unsafe)
    axes[1, 0] = ax3
    contour = ax3.contourf(np.meshgrid(x, y)[0], np.meshgrid(x, y)[1], z2, levels=10, cmap='turbo')
    ax3.contour(np.meshgrid(x, y)[0], np.meshgrid(x, y)[1], z2, levels=[0], colors='black', linewidths=2)
    ax3.set_title('best-case CBF Derivative Contour')
    plt.colorbar(contour, ax=ax3)
    plt.savefig('best-case_CBF_Derivative_Contour.png', dpi=300)
    
    # Plot 4: Heatmap of best-case CBF derivative
    _, ax4 = plot_env(X, X_unsafe)
    axes[1, 1] = ax4
    heatmap = ax4.imshow(z2, extent=[x.min(), x.max(), y.min(), y.max()], 
                        origin='lower', aspect='auto', cmap='turbo')
    ax4.set_title('best-case CBF Derivative Heatmap')
    plt.colorbar(heatmap, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('best-case_CBF_Derivative_Heatmap.png', dpi=300)
    plt.show()
    
    return fig

# Define your neural network model
class CBFModel(nn.Module):
    def __init__(self):
        super(CBFModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Load your trained model here
def load_model(model_path):
    model = CBFModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Example usage
if __name__ == "__main__":
    # Load your model
    model = load_model("car_naive_model_1.0_0.0_0.1_pgd_relu_20.pt")
    
    # For demonstration, create a dummy model
    model = CBFModel()
    
    # Visualize CBF
    visualize_cbf(
        model,
        alpha=0.0,
        theta=0,
        v=1,
        w=1,
        pgd_lr=10,
        pgd_num_iter=10
    )
