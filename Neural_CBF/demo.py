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
import random
import pickle
class Hyperrectangle:
    def __init__(self, low, high, npy=True):
        if npy:
            self.low = np.array(low, dtype=np.float32)
            self.high = np.array(high, dtype=np.float32)
        else:
            self.low = torch.tensor(low, dtype=torch.float32).to(device)
            self.high = torch.tensor(high, dtype=torch.float32).to(device)
        self.dim = len(low)
        self.npy = npy

    def contains(self, point):
        """Check if a point is inside the hyperrectangle."""
        assert self.npy
        low = np.array(self.low, dtype=np.float32)
        high = np.array(self.high, dtype=np.float32)
        return np.all(point >= low) and np.all(point <= high)

    def not_contains(self, point):
        """Check if a point is outside the hyperrectangle."""
        return not self.contains(point)

    def vertices_list(self):
        """Generate all vertices of the hyperrectangle"""
        n = len(self.low)
        vertices = []

        # Generate all combinations of low and high values
        for i in range(2 ** n):
            vertex = torch.zeros_like(self.low)
            for j in range(n):
                if (i >> j) & 1:
                    vertex[j] = self.high[j]
                else:
                    vertex[j] = self.low[j]
            vertices.append(vertex)

        return vertices


# Dubins Car model
def random_point_in_hyperrectangle(hyperrectangle, non_admissible_area=None, q=False):
    """Generate a random point inside a hyperrectangle and outside a non-admissible area if specified."""
    dimensions = hyperrectangle.dim
    random_point = np.zeros(dimensions)

    for i in range(dimensions):
        random_point[i] = random.random() * (hyperrectangle.high[i] - hyperrectangle.low[i]) + hyperrectangle.low[i]

    if q:
        # Handle quaternions for quadrotor model (not needed for DubinsCar)
        pass

    if non_admissible_area is None:
        return random_point, True

    if non_admissible_area.not_contains(random_point):
        return random_point, True

    return random_point, False


def generate_xref(dmodel, x_0, dt, T, X, X_unsafe, U, max_u=10000, euler=False):
    """Generate a reference trajectory starting from x_0."""
    n_steps = int(np.floor(T / dt))
    Uref = []
    Xref = [x_0]

    for i in range(n_steps):
        u = None
        x = Xref[-1]
        x_next = None
        feasible = False

        for j in range(max_u):
            u, _ = random_point_in_hyperrectangle(U)

            if euler:
                # Use scipy's ODE solver
                def f(t, x):
                    return dmodel.dynamics(x, u)

                sol = solve_ivp(f, [0, dt], x, method='RK45', rtol=1e-8, atol=1e-8)
                x_next = sol.y[:, -1]
            else:
                # Use discrete dynamics
                x_next = dmodel.discrete_dynamics(x, u, 0.0, dt)

            if X_unsafe.not_contains(x_next) and X.contains(x_next):
                feasible = True
                break

        if not feasible:
            if len(Uref) == 1:
                return Xref, Uref
            if len(Xref) == 1:
                return Xref, Uref

            Xref.pop()
            Uref.pop()
            continue

        Xref.append(x_next)
        Uref.append(u)

    return Xref, Uref


def generate_random_traj(dmodel, num, dt, T, X, X_unsafe, U, q=False, euler=False):
    """Generate multiple random trajectories."""
    Xrefs = []
    Urefs = []

    for i in tqdm(range(num), desc="Generating trajectories"):
        x_0 = None
        while True:
            x_0, safe_flag = random_point_in_hyperrectangle(X, X_unsafe, q=q)
            if safe_flag:
                break

        Xref, Uref = generate_xref(dmodel, x_0, dt, T, X, X_unsafe, U, euler=euler)
        Xrefs.append(Xref)
        Urefs.append(Uref)

    return Xrefs, Urefs


def build_dataset(name, Xrefs, Urefs, X, X_unsafe, U, n_ignore=50, q=False):
    """Build and save the dataset for training and testing."""
    data = []
    # Add safe trajectories
    for k in range(len(Xrefs)):
        if len(Urefs[k]) < n_ignore + 1:
            continue

        for i in range(len(Urefs[k]) - n_ignore):
            # Safe and persistently feasible
            data.append([Xrefs[k][i], Urefs[k][i], [True]])

    n_safe = int(np.floor(len(data) * 0.8))
    # Add unsafe points
    for i in range(n_safe):
        random_x0, safe_flag = random_point_in_hyperrectangle(X_unsafe, X_unsafe, q=q)
        random_u0, _ = random_point_in_hyperrectangle(U)
        assert safe_flag == False
        data.append([random_x0, random_u0, [safe_flag]])

    # Convert to numpy array
    X_data = np.array([d[0] for d in data])
    U_data = np.array([d[1] for d in data])
    Y_data = np.array([d[2] for d in data])

    # Shuffle the data
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    X_data = X_data[indices]
    U_data = U_data[indices]
    Y_data = Y_data[indices]

    # Combine into one list of tuples for easier handling
    combined_data = [(X_data[i], U_data[i], Y_data[i]) for i in range(len(data))]
    # Split into training and test sets
    test_size = min(10000, len(combined_data) // 5)
    training_data = combined_data[:-test_size]
    test_data = combined_data[-test_size:]

    # Save to files
    with open(f"{name}_training_data.pkl", "wb") as f:
        pickle.dump(training_data, f)

    with open(f"{name}_test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    print(f"Saved {len(training_data)} training samples and {len(test_data)} test samples")

    return training_data, test_data


def plot_trajectories(Xrefs, X_unsafe, X, title="Dubins Car Trajectories", max_traj=np.inf):
    """Plot some of the generated trajectories."""
    plt.figure(figsize=(10, 8))

    # Plot a subset of trajectories to avoid cluttering
    for i in range(min(len(Xrefs), max_traj)):
        x_coords = [x[0] for x in Xrefs[i]]
        y_coords = [x[1] for x in Xrefs[i]]
        plt.plot(x_coords, y_coords, 'b-', alpha=0.3)

        # Mark the starting point
        plt.plot(x_coords[0], y_coords[0], 'go')

    # Plot the unsafe region
    unsafe_x = [X_unsafe.low[0], X_unsafe.high[0], X_unsafe.high[0], X_unsafe.low[0], X_unsafe.low[0]]
    unsafe_y = [X_unsafe.low[1], X_unsafe.low[1], X_unsafe.high[1], X_unsafe.high[1], X_unsafe.low[1]]
    plt.plot(unsafe_x, unsafe_y, 'r-', linewidth=2)

    # Plot the domain
    domain_x = [X.low[0], X.high[0], X.high[0], X.low[0], X.low[0]]
    domain_y = [X.low[1], X.low[1], X.high[1], X.high[1], X.low[1]]
    plt.plot(domain_x, domain_y, 'k-', linewidth=2)

    plt.title(title)
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()




def demo():
    pass


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    demo()
