import numpy as np
import matplotlib.pyplot as plt

# Function to compute throughputs S
def compute_S(p):
    p0, p1, p2 = p
    q0 = 1 - p0
    q1 = 1 - p1
    q2 = 1 - p2

    # Numerators
    numerator_S0 = p0 * q1 * q2 + p0 * q1 * p2 + q1 * p0 * p2
    numerator_S1 = q0 * p1 * q2
    numerator_S2 = p0 * q1 * p2 + q0 * q1 * p2 + p0 * q1 * p2

    # Denominator
    D = 1 + q1 * p2 + q1 * p0 + q0 * p1 + p0 * p1 + q1 * p0 * p2

    # Assuming T_star = 2 for simplicity
    T_star = 2

    S0 = T_star * (numerator_S0 / D)
    S1 = T_star * (numerator_S1 / D)
    S2 = T_star * (numerator_S2 / D)

    return np.array([S0, S1, S2])

# Function to compute gradient of log(S) with respect to p
def compute_gradient_log_S(p, delta=1e-6):
    n = len(p)
    grad_log_S = np.zeros((n, n))  # grad_log_S[i, j] = d(log(S_i))/dp_j

    for j in range(n):
        p_forward = p.copy()
        p_backward = p.copy()
        p_forward[j] += delta
        p_backward[j] -= delta

        # Ensure p values remain within [0, 1]
        p_forward[j] = min(p_forward[j], 1)
        p_backward[j] = max(p_backward[j], 0)

        S_forward = compute_S(p_forward)
        S_backward = compute_S(p_backward)

        # Compute log of S values, adding delta to prevent log(0)
        log_S_forward = np.log(S_forward + delta)
        log_S_backward = np.log(S_backward + delta)

        grad_log_S[:, j] = (log_S_forward - log_S_backward) / (2 * delta)

    return grad_log_S  # Shape: (n, n)

# Initialize parameters
p_init = np.array([0.97, 0.01, 0.05])
alpha_init = np.array([0.6, 0.6, 0.3])
eta_p = 0.01
max_iter_short = 200

# Store values for plotting
p_values_short = []
S_values_short = []
J_values_short = []

# Optimization loop for shorter iteration count
p = p_init.copy()
alpha = alpha_init.copy()
for k in range(max_iter_short):
    # Compute S and gradients
    S = compute_S(p)
    grad_S = compute_gradient_log_S(p)

    # Gradient of J with respect to p
    grad_J_p = np.dot(alpha, grad_S)

    # Update p and project onto [0, 1]
    p += eta_p * grad_J_p
    p = np.clip(p, 0, 1)

    # Calculate cost J and store values every 5 iterations for this run
    if k % 5 == 0:
        J = np.dot(alpha, np.log(S))
        p_values_short.append(p.copy())
        S_values_short.append(S.copy())
        J_values_short.append(J)

# Convert lists to arrays for easier plotting
p_values_short = np.array(p_values_short)
S_values_short = np.array(S_values_short)
J_values_short = np.array(J_values_short)

# Generate the plots in a row
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)

# Plotting Transmit Probabilities (p0, p1, p2)
axes[0].plot(np.arange(0, max_iter_short, 5), p_values_short[:, 0], label="p0", linewidth=1.5)
axes[0].plot(np.arange(0, max_iter_short, 5), p_values_short[:, 1], label="p1", linewidth=1.5)
axes[0].plot(np.arange(0, max_iter_short, 5), p_values_short[:, 2], label="p2", linewidth=1.5)
axes[0].set_title("Transmit Probabilities (p) vs Iterations")
axes[0].set_ylabel("Probability")
axes[0].set_xlabel("Iterations")
axes[0].legend(loc="upper right", fontsize=8, frameon=False)
axes[0].grid(alpha=0.4)

# Plotting Throughputs (S0, S1, S2)
axes[1].plot(np.arange(0, max_iter_short, 5), S_values_short[:, 0], label="S0", linewidth=1.5)
axes[1].plot(np.arange(0, max_iter_short, 5), S_values_short[:, 1], label="S1", linewidth=1.5)
axes[1].plot(np.arange(0, max_iter_short, 5), S_values_short[:, 2], label="S2", linewidth=1.5)
axes[1].set_title("Throughputs (S) vs Iterations")
axes[1].set_ylabel("Throughput")
axes[1].set_xlabel("Iterations")
axes[1].legend(loc="upper right", fontsize=8, frameon=False)
axes[1].grid(alpha=0.4)

# Plotting Total Cost (J)
axes[2].plot(np.arange(0, max_iter_short, 5), J_values_short, label="J", color='purple', linewidth=1.5)
axes[2].set_title("Total Cost (J) vs Iterations")
axes[2].set_xlabel("Iterations")
axes[2].set_ylabel("Total Cost J")
axes[2].legend(loc="lower right", fontsize=8, frameon=False)
axes[2].grid(alpha=0.4)

# Adjust layout for a tight fit
plt.tight_layout()
plt.show()
