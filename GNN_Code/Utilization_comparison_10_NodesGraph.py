
import time
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
P_INIT = np.array([0.15, 0.20, 0.25, 0.18, 0.22, 0.12, 0.30, 0.10, 0.16, 0.14])
   # Initial p; adjust size to match ADJ_MATRIX
ALPHA = np.array([0.8, 1.0, 0.9, 1.1, 0.95, 1.0, 0.85, 0.9, 1.0, 1.05])
      # Utility weights (same length as p)
LEARNING_RATE = 0.01
MAX_ITER = 250
LOG_EVERY = 5
UTILITY_MODE = 'log'  # 'log' or 'linear'
EPS = 1e-9
T_STAR = 2           # Packet duration (T)
SIGMA = 1           # Idle slot duration (fixed at 1 in this discrete model)
OPTIMIZER_TYPE = 'sgd'  # 'sgd' or 'adam'

# Model settings
MODEL_PATH = "2_DGCN.pt"
INPUT_DIM = 1
HIDDEN_DIMS = [64, 64, 64, 64, 64, 64, 64]
OUTPUT_DIM = 1

# -----------------------------------------------------------
# Choose topology here
# -----------------------------------------------------------
# Example A: 3-node chain 0--1--2  (matches your optimizer defaults)
ADJ_MATRIX = [
    # 0  1  2  3  4  5  6  7  8  9
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # Node 0 connected to 1,3
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # Node 1 connected to 0,2,4
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  # Node 2 connected to 1,3,5
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Node 3 connected to 0,2,6
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],  # Node 4 connected to 1,5,7
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],  # Node 5 connected to 2,4,8
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  # Node 6 connected to 3,7,9
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # Node 7 connected to 4,6
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Node 8 connected to 5,9
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # Node 9 connected to 6,8
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------------------------------
# Markov-chain engine (general n, T)
# -----------------------------------------------------------
def convert_num_to_state(n, T, num):
    # base-T digits (little-endian: index 0 is least-significant)
    return [(num // (T**i)) % T for i in range(n)]

def convert_state_to_num(n, T, state):
    total, mult = 0, 1
    for j in range(n):
        total += mult * state[j]
        mult *= T
    return total

def convert_num_to_trans(n, num):
    # n-bit vector (little-endian)
    return [(num >> i) & 1 for i in range(n)]

def transition_from_state(n, T, G, probs, st):

    num_states = T ** n
    num_trans = 2 ** n
    tr_probs = [0.0] * num_states
    tot_reward = [0.0] * n

    for mask in range(num_trans):
        trans = convert_num_to_trans(n, mask)  # 0/1 attempt decision
        valid = True
        pr = 1.0

        # Probability factorization & feasibility
        for j in range(n):
            if trans[j] == 1:
                # must be idle AND all neighbors idle, else invalid (CSMA sensing)
                if st[j] > 0:
                    valid = False; break
                for k in range(n):
                    if G[j][k] == 1 and st[k] > 0:
                        valid = False
                        break
                if not valid: break
                pr *= probs[j]
            else:
                # if idle and neighbors idle, we explicitly multiply (1-p_j);
                # if busy or neighbor busy, the node cannot attempt anyway (no factor).
                if st[j] == 0:
                    for k in range(n):
                        if G[j][k] == 1 and st[k] > 0:
                            break
                    else:
                        pr *= (1 - probs[j])

        if not valid:
            continue

        # Apply the transition
        new_state = [0] * n
        for j in range(n):
            if trans[j] == 1:
                new_state[j] = T - 1
            elif st[j] > 0:
                new_state[j] = st[j] - 1
            else:
                new_state[j] = 0

        # Rewards: successful starts (no simultaneous neighbor attempt)
        for j in range(n):
            if trans[j] == 1:
                for k in range(n):
                    if G[j][k] == 1 and trans[k] == 1:
                        break
                else:
                    tot_reward[j] += pr

        idx = convert_state_to_num(n, T, new_state)
        tr_probs[idx] += pr

    return tr_probs, tot_reward

def build_transition_matrix_and_rewards(G, probs, T):
    n = len(probs)
    num_states = T ** n
    P = np.zeros((num_states, num_states), dtype=np.float64)
    R = np.zeros((num_states, n), dtype=np.float64)  # per-state success-prob vector

    for s in range(num_states):
        st = convert_num_to_state(n, T, s)
        tr, rew = transition_from_state(n, T, G, probs, st)
        P[s, :] = tr
        R[s, :] = np.asarray(rew)  # success-probability for each node from state s

    return P, R

def stationary_distribution(P, tol=1e-12, max_iter=1_000_000):
    """
    Power iteration on P^T to get left eigenvector for eigenvalue 1 (Markov stationary).
    More stable than dense eig for moderate sizes.
    """
    num_states = P.shape[0]
    pi = np.ones(num_states, dtype=np.float64) / num_states
    PT = P.T
    for _ in range(max_iter):
        new = PT @ pi
        # normalize to sum 1
        new_sum = new.sum()
        if new_sum == 0:
            # degenerate (shouldn't happen if P is stochastic)
            new = np.ones_like(new) / num_states
        else:
            new = new / new_sum
        if np.linalg.norm(new - pi, 1) < tol:
            return new
        pi = new
    return pi  # fallback

def compute_mc_throughput(p, G, T):
    """
    Compute per-node saturation throughput via Markov chain:
      - Build P (transition) and per-step success reward R
      - Stationary pi
      - Throughput_i = T * E_pi[success_start_i]
    """
    n = len(p)
    P, R = build_transition_matrix_and_rewards(G, p, T)
    pi = stationary_distribution(P)
    # expected successes per step per node = pi^T R
    succ = pi @ R
    return T * succ  # each success occupies T slots

# -----------------------------------------------------------
# GNN helpers (unchanged)
# -----------------------------------------------------------
def load_gnn_model():
    """Load the trained DGCN model."""
    from GNN_Architecture import DGCNConv
    model = DGCNConv(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM,
                     dropout=0.5, mlp_layers=[64, 32]).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def build_edge_index(adj):
    edges = [[i, j] for i, row in enumerate(adj) for j, val in enumerate(row) if val == 1]
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

def utility_function(theta, mode='log', eps=1e-9):
    return torch.log(theta + eps) if mode == 'log' else theta

# -----------------------------------------------------------
# Optimization (theory = MC now)
# -----------------------------------------------------------
def compute_theory_gradient_via_fd(p, alpha, utility_mode='log', eps=1e-9, delta=1e-6):
    """Finite-difference gradient of J wrt p using MC throughput as the oracle."""
    n = len(p)
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p_plus  = p.copy();  p_plus[i]  = min(p_plus[i]  + delta, 1.0)
        p_minus = p.copy();  p_minus[i] = max(p_minus[i] - delta, 0.0)

        S_plus  = compute_mc_throughput(p_plus,  ADJ_MATRIX, T_STAR)
        S_minus = compute_mc_throughput(p_minus, ADJ_MATRIX, T_STAR)

        if utility_mode == 'log':
            J_plus  = np.dot(alpha, np.log(S_plus  + eps))
            J_minus = np.dot(alpha, np.log(S_minus + eps))
        else:
            J_plus  = np.dot(alpha, S_plus)
            J_minus = np.dot(alpha, S_minus)

        grad[i] = (J_plus - J_minus) / (2 * delta)
    return grad

def optimize_theory_mc(p_init, alpha, max_iter, lr, optimizer_type='sgd'):
    """Optimize using MC-based throughput (no closed form)."""
    p = torch.tensor(p_init, dtype=torch.float32, requires_grad=True)

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam([p], lr=lr)
    else:
        optimizer = torch.optim.SGD([p], lr=lr)

    history = {'p': [], 'S': [], 'J': []}

    for k in range(max_iter):
        optimizer.zero_grad()

        # finite-difference gradient via MC oracle
        grad_np = compute_theory_gradient_via_fd(
            p.detach().numpy(), alpha, UTILITY_MODE, EPS
        )
        p.grad = torch.tensor(-grad_np, dtype=torch.float32)  # maximize

        optimizer.step()

        with torch.no_grad():
            p.clamp_(0.0, 1.0)

        if k % LOG_EVERY == 0:
            p_np = p.detach().numpy()
            S_np = compute_mc_throughput(p_np, ADJ_MATRIX, T=T_STAR)
            J = np.dot(alpha, np.log(S_np + EPS)) if UTILITY_MODE == 'log' else np.dot(alpha, S_np)

            history['p'].append(p_np.copy())
            history['S'].append(S_np.copy())
            history['J'].append(J)

            if k % 20 == 0:
                print(f"[Theory(MC) {k:03d}] J={J:.6f} | p={np.round(p_np, 5)}")

    return history

def optimize_gnn(model, p_init, alpha, max_iter, lr, optimizer_type='sgd'):
    """Optimize using GNN model (unchanged)."""
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

        data = Data(x=p.unsqueeze(1), edge_index=edge_index).to(device)
        theta_pred = model(data).squeeze(-1)

        J = (alpha_tensor * utility_function(theta_pred, UTILITY_MODE, EPS)).sum()
        loss = -J
        loss.backward()

        optimizer.step()

        with torch.no_grad():
            p.clamp_(0.0, 1.0)

        if k % LOG_EVERY == 0:
            p_np = p.detach().cpu().numpy()
            S_np = theta_pred.detach().cpu().numpy()
            J_val = J.item()

            history['p'].append(p_np.copy())
            history['S'].append(S_np.copy())
            history['J'].append(J_val)

            if k % 20 == 0:
                print(f"[GNN       {k:03d}] J={J_val:.6f} | p={np.round(p_np, 5)}")

    return history

# -----------------------------------------------------------
# Visualization & summary
# -----------------------------------------------------------
def plot_comparison(history_theory, history_gnn):
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    p_theory = np.array(history_theory['p'])
    S_theory = np.array(history_theory['S'])
    J_theory = np.array(history_theory['J'])

    p_gnn = np.array(history_gnn['p'])
    S_gnn = np.array(history_gnn['S'])
    J_gnn = np.array(history_gnn['J'])

    iters = np.arange(0, MAX_ITER, LOG_EVERY)
    colors = [f"C{i}" for i in range(S_theory.shape[1])]

    # 1. p trajectories
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(p_theory.shape[1]):
        ax1.plot(iters, p_theory[:, i], '-',  color=colors[i], label=f'p{i} (MC)', linewidth=2)
        ax1.plot(iters, p_gnn[:, i],   '--', color=colors[i], label=f'p{i} (D-GCN)', linewidth=2)
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('Probability'); ax1.legend(ncol=3, fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # 2. Throughput trajectories
    ax2 = fig.add_subplot(gs[1, :])
    for i in range(S_theory.shape[1]):
        ax2.plot(iters, S_theory[:, i], '-',  color=colors[i], label=f'Θ{i} (MC)', linewidth=2)
        ax2.plot(iters, S_gnn[:, i],   '--', color=colors[i], label=f'Θ{i} (D-GCN)', linewidth=2)
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Throughput'); ax2.legend(ncol=3, fontsize=8); ax2.grid(True, alpha=0.3)

    # 3. Objective
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(iters, J_theory, 'k-', label='MC', linewidth=2)
    ax3.plot(iters, J_gnn,    'r--', label='D-GCN', linewidth=2)
    ax3.set_xlabel('Iteration'); ax3.set_ylabel('J'); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Final comparison printout
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'':15} {'MC (Theory)':>20} {'D-GCN':>20}")
    print("-"*60)
    print("p (final):")
    for i in range(p_theory.shape[1]):
        print(f"  p{i}:        {p_theory[-1, i]:20.6f} {p_gnn[-1, i]:20.6f}")
    print("-"*60)
    print("Throughput (final):")
    for i in range(S_theory.shape[1]):
        print(f"  Θ{i}:        {S_theory[-1, i]:20.6f} {S_gnn[-1, i]:20.6f}")
    print("-"*60)
    print(f"J (final):     {J_theory[-1]:20.6f} {J_gnn[-1]:20.6f}")
    print("-"*60)

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    n = len(ADJ_MATRIX)
    assert len(P_INIT) == n and len(ALPHA) == n, \
        "P_INIT and ALPHA must match the number of nodes in ADJ_MATRIX."

    print("Starting fair comparison: MC (Markov Chain) vs GNN")
    print(f"Settings: Optimizer={OPTIMIZER_TYPE}, LR={LEARNING_RATE}, Iterations={MAX_ITER}")
    print(f"Initial p: {P_INIT}")
    print(f"Alpha weights: {ALPHA}")
    print(f"Packet duration T: {T_STAR}")
    print()

    # Load GNN
    print("Loading GNN model...")
    model = load_gnn_model()
    print("Model loaded successfully!")

    # Initial predictions
    print("\nInitial throughput predictions:")
    S_theory_init = compute_mc_throughput(P_INIT, ADJ_MATRIX, T=T_STAR)
    print(f"MC (exact): {np.round(S_theory_init, 6)}")

    with torch.no_grad():
        edge_index = build_edge_index(ADJ_MATRIX)
        data = Data(
            x=torch.tensor(P_INIT, dtype=torch.float32, device=device).unsqueeze(1),
            edge_index=edge_index
        ).to(device)
        S_gnn_init = model(data).squeeze(-1).cpu().numpy()
        print(f"GNN:        {np.round(S_gnn_init, 6)}")

    print("\nStarting optimization...")
    print("-"*60)

    # MC optimization
    history_theory = optimize_theory_mc(P_INIT, ALPHA, MAX_ITER, LEARNING_RATE, OPTIMIZER_TYPE)
    print("\nMC optimization complete!")

    # GNN optimization
    history_gnn = optimize_gnn(model, P_INIT, ALPHA, MAX_ITER, LEARNING_RATE, OPTIMIZER_TYPE)
    print("GNN optimization complete!")

    # Plots & summary
    plot_comparison(history_theory, history_gnn)

if __name__ == "__main__":
    main()
