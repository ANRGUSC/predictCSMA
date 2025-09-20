#!/usr/bin/env python3
"""
Fair comparison between theoretical Markov chain p-CSMA model and trained GNN model
for network utility maximization with throughput prediction.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
# Common settings for fair comparison
P_INIT = np.array([0.97, 0.01, 0.05])
ALPHA = np.array([0.6, 0.6, 0.3])
LEARNING_RATE = 0.01
MAX_ITER = 250
LOG_EVERY = 5
UTILITY_MODE = 'log'  # 'log' or 'linear'
EPS = 1e-9
T_STAR = 2  # Packet duration
OPTIMIZER_TYPE = 'sgd'  # 'sgd' or 'adam' - use same for both!

# Model settings
MODEL_PATH = "211_bestenhanced.pt"
INPUT_DIM = 1
HIDDEN_DIMS = [64, 64, 64, 64, 64, 64, 64]
OUTPUT_DIM = 1

# 3-node chain topology: 0--1--2
ADJ_MATRIX = [
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
]

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------------------------------
# Theoretical Model Functions
# -----------------------------------------------------------
def compute_theoretical_throughput(p, T=2):
    """
    Compute exact throughput for 3-node chain using Markov chain analysis.
    This assumes the specific topology: 0--1--2
    """
    p0, p1, p2 = p
    q0 = 1 - p0
    q1 = 1 - p1
    q2 = 1 - p2

    # Based on Markov chain analysis for this specific topology
    # Node 0 and 2 interfere through node 1
    numerator_S0 = p0 * q1 * q2 + p0 * q1 * p2 + q1 * p0 * p2
    numerator_S1 = q0 * p1 * q2
    numerator_S2 = p0 * q1 * p2 + q0 * q1 * p2 + p0 * q1 * p2

    # Common denominator
    D = 1 + q1 * p2 + q1 * p0 + q0 * p1 + p0 * p1 + q1 * p0 * p2

    S0 = T * (numerator_S0 / D)
    S1 = T * (numerator_S1 / D)
    S2 = T * (numerator_S2 / D)

    return np.array([S0, S1, S2])

def compute_theoretical_gradient(p, alpha, utility_mode='log', eps=1e-9, delta=1e-6):
    """
    Compute gradient of objective J w.r.t. p using finite differences.
    """
    n = len(p)
    grad = np.zeros(n)
    
    for i in range(n):
        # Forward difference
        p_plus = p.copy()
        p_plus[i] = min(p_plus[i] + delta, 1.0)
        S_plus = compute_theoretical_throughput(p_plus, T=T_STAR)
        
        # Backward difference
        p_minus = p.copy()
        p_minus[i] = max(p_minus[i] - delta, 0.0)
        S_minus = compute_theoretical_throughput(p_minus, T=T_STAR)
        
        # Compute utilities
        if utility_mode == 'log':
            U_plus = np.log(S_plus + eps)
            U_minus = np.log(S_minus + eps)
        else:
            U_plus = S_plus
            U_minus = S_minus
        
        # Compute objective values
        J_plus = np.dot(alpha, U_plus)
        J_minus = np.dot(alpha, U_minus)
        
        # Gradient
        grad[i] = (J_plus - J_minus) / (2 * delta)
    
    return grad

# -----------------------------------------------------------
# GNN Model Functions
# -----------------------------------------------------------
def load_gnn_model():
    """Load the trained DGCN model."""
    from GNN_Architecture import DGCNConv
    
    model = DGCNConv(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM,
                            dropout=0.5, mlp_layers=[64, 32]).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    # Freeze weights
    for p in model.parameters():
        p.requires_grad = False
    
    return model

def build_edge_index(adj):
    """Convert adjacency matrix to PyG edge_index tensor."""
    edges = [[i, j] for i, row in enumerate(adj) 
             for j, val in enumerate(row) if val == 1]
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

def utility_function(theta, mode='log', eps=1e-9):
    """Compute utility from throughput."""
    if mode == 'log':
        return torch.log(theta + eps)
    else:
        return theta

# -----------------------------------------------------------
# Optimization Functions
# -----------------------------------------------------------
def optimize_theoretical(p_init, alpha, max_iter, lr, optimizer_type='sgd'):
    """Optimize using theoretical model."""
    p = torch.tensor(p_init, dtype=torch.float32, requires_grad=True)
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam([p], lr=lr)
    else:
        optimizer = torch.optim.SGD([p], lr=lr)
    
    history = {'p': [], 'S': [], 'J': []}
    
    for k in range(max_iter):
        optimizer.zero_grad()
        
        # Compute gradient using finite differences
        grad_np = compute_theoretical_gradient(
            p.detach().numpy(), alpha, UTILITY_MODE, EPS
        )
        
        # Set gradient (negative because we want to maximize)
        p.grad = torch.tensor(-grad_np, dtype=torch.float32)
        
        optimizer.step()
        
        # Project to [0, 1]
        with torch.no_grad():
            p.clamp_(0.0, 1.0)
        
        # Log progress
        if k % LOG_EVERY == 0:
            p_np = p.detach().numpy()
            S_np = compute_theoretical_throughput(p_np, T=T_STAR)
            
            if UTILITY_MODE == 'log':
                J = np.dot(alpha, np.log(S_np + EPS))
            else:
                J = np.dot(alpha, S_np)
            
            history['p'].append(p_np.copy())
            history['S'].append(S_np.copy())
            history['J'].append(J)
            
            if k % 20 == 0:
                print(f"[Theory {k:03d}] J={J:.6f} | p={np.round(p_np, 5)}")
    
    return history

def optimize_gnn(model, p_init, alpha, max_iter, lr, optimizer_type='sgd'):
    """Optimize using GNN model."""
    edge_index = build_edge_index(ADJ_MATRIX)
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device)
    
    p = torch.tensor(p_init, dtype=torch.float32, device=device, requires_grad=True)
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam([p], lr=lr)
    else:
        optimizer = torch.optim.SGD([p], lr=lr)
    
    history = {'p': [], 'S': [], 'J': []}
    
    for k in range(max_iter):
        optimizer.zero_grad()
        
        # Create data object
        data = Data(x=p.unsqueeze(1), edge_index=edge_index).to(device)
        
        # Forward pass
        theta_pred = model(data).squeeze(-1)
        
        # Compute objective
        J = (alpha_tensor * utility_function(theta_pred, UTILITY_MODE, EPS)).sum()
        
        # Backward pass (negative for maximization)
        loss = -J
        loss.backward()
        
        optimizer.step()
        
        # Project to [0, 1]
        with torch.no_grad():
            p.clamp_(0.0, 1.0)
        
        # Log progress
        if k % LOG_EVERY == 0:
            p_np = p.detach().cpu().numpy()
            S_np = theta_pred.detach().cpu().numpy()
            J_val = J.item()
            
            history['p'].append(p_np.copy())
            history['S'].append(S_np.copy())
            history['J'].append(J_val)
            
            if k % 20 == 0:
                print(f"[GNN    {k:03d}] J={J_val:.6f} | p={np.round(p_np, 5)}")
    
    return history

# -----------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------
def plot_comparison(history_theory, history_gnn):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Convert histories to arrays
    p_theory = np.array(history_theory['p'])
    S_theory = np.array(history_theory['S'])
    J_theory = np.array(history_theory['J'])
    
    p_gnn = np.array(history_gnn['p'])
    S_gnn = np.array(history_gnn['S'])
    J_gnn = np.array(history_gnn['J'])
    
    iters = np.arange(0, MAX_ITER, LOG_EVERY)
    
    # Colors for each node
    colors = ['blue', 'orange', 'green']
    
    # 1. Transmission Probabilities Comparison
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(3):
        ax1.plot(iters, p_theory[:, i], '-', color=colors[i], 
                label=f'p{i} (Markov (Exact))', linewidth=2)
        ax1.plot(iters, p_gnn[:, i], '--', color=colors[i], 
                label=f'p{i} (D-GCN)', linewidth=2)
    # ax1.set_title('Transmission Probabilities: Theory vs GNN', fontsize=14)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Probability')
    ax1.legend(ncol=3, fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # 2. Throughput Comparison
    ax2 = fig.add_subplot(gs[1, :])
    for i in range(3):
        ax2.plot(iters, S_theory[:, i], '-', color=colors[i], 
                label=f'Θ{i} (Markov (Exact))', linewidth=2)
        ax2.plot(iters, S_gnn[:, i], '--', color=colors[i], 
                label=f'Θ{i} (D-GCN)', linewidth=2)
    # ax2.set_title('Throughput: Theory vs GNN', fontsize=14)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Throughput')
    ax2.legend(ncol=3, fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # 3. Objective Function
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(iters, J_theory, 'k-', label='Markov (Exact)', linewidth=2)
    ax3.plot(iters, J_gnn, 'r--', label='D-GCN', linewidth=2)
    # ax3.set_title('Objective Function J', fontsize=14)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('J')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # # 4. Relative Error in p
    # ax4 = fig.add_subplot(gs[2, 1])
    rel_error_p = np.abs(p_gnn - p_theory) / (p_theory + 1e-6) * 100
    # for i in range(3):
    #     ax4.plot(iters, rel_error_p[:, i], '-', color=colors[i], 
    #             label=f'p{i}', linewidth=2)
    # ax4.set_title('Relative Error in p (%)', fontsize=14)
    # ax4.set_xlabel('Iteration')
    # ax4.set_ylabel('Error (%)')
    # ax4.legend(fontsize=10)
    # ax4.grid(True, alpha=0.3)
    # ax4.set_yscale('log')
    
    # # 5. Relative Error in Throughput
    # ax5 = fig.add_subplot(gs[2, 2])
    rel_error_S = np.abs(S_gnn - S_theory) / (S_theory + 1e-6) * 100
    # for i in range(3):
    #     ax5.plot(iters, rel_error_S[:, i], '-', color=colors[i], 
    #             label=f'Θ{i}', linewidth=2)
    # ax5.set_title('Relative Error in Throughput (%)', fontsize=14)
    # ax5.set_xlabel('Iteration')
    # ax5.set_ylabel('Error (%)')
    # ax5.legend(fontsize=10)
    # ax5.grid(True, alpha=0.3)
    # ax5.set_yscale('log')
    
    # plt.suptitle(f'Theory vs GNN Comparison (Optimizer: {OPTIMIZER_TYPE.upper()}, LR: {LEARNING_RATE})', 
    #              fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'':15} {'Theory':>20} {'GNN':>20}")
    print("-"*60)
    
    # Final p values
    for i in range(3):
        print(f"p{i}:            {p_theory[-1, i]:20.6f} {p_gnn[-1, i]:20.6f}")
    
    print("-"*60)
    
    # Final throughput
    for i in range(3):
        print(f"Θ{i}:            {S_theory[-1, i]:20.6f} {S_gnn[-1, i]:20.6f}")
    
    print("-"*60)
    
    # Final objective
    print(f"J:             {J_theory[-1]:20.6f} {J_gnn[-1]:20.6f}")
    
    print("-"*60)
    
    # Average relative errors
    avg_error_p = np.mean(rel_error_p[-1])
    avg_error_S = np.mean(rel_error_S[-1])
    print(f"Avg Error p:   {avg_error_p:20.2f}%")
    print(f"Avg Error Θ:   {avg_error_S:20.2f}%")

# -----------------------------------------------------------
# Main Execution
# -----------------------------------------------------------
def main():
    print("Starting fair comparison: Theory vs GNN")
    print(f"Settings: Optimizer={OPTIMIZER_TYPE}, LR={LEARNING_RATE}, Iterations={MAX_ITER}")
    print(f"Initial p: {P_INIT}")
    print(f"Alpha weights: {ALPHA}")
    print(f"Packet duration T: {T_STAR}")
    print()
    
    # Load GNN model
    print("Loading GNN model...")
    model = load_gnn_model()
    print("Model loaded successfully!")
    
    # Test initial predictions
    print("\nInitial throughput predictions:")
    S_theory_init = compute_theoretical_throughput(P_INIT, T=T_STAR)
    print(f"Theory: {np.round(S_theory_init, 6)}")
    
    with torch.no_grad():
        edge_index = build_edge_index(ADJ_MATRIX)
        data = Data(
            x=torch.tensor(P_INIT, dtype=torch.float32, device=device).unsqueeze(1),
            edge_index=edge_index
        ).to(device)
        S_gnn_init = model(data).squeeze(-1).cpu().numpy()
        print(f"GNN:    {np.round(S_gnn_init, 6)}")
    
    print("\nStarting optimization...")
    print("-"*60)
    
    # Run optimizations
    history_theory = optimize_theoretical(P_INIT, ALPHA, MAX_ITER, LEARNING_RATE, OPTIMIZER_TYPE)
    print("\nTheory optimization complete!")
    
    history_gnn = optimize_gnn(model, P_INIT, ALPHA, MAX_ITER, LEARNING_RATE, OPTIMIZER_TYPE)
    print("GNN optimization complete!")
    
    # Plot results
    plot_comparison(history_theory, history_gnn)

if __name__ == "__main__":
    main()