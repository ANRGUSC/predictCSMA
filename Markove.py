import numpy as np

# Example input:
# Transmission probability for each node (11 nodes)
transmission_prob = [0.2, 0.1, 0.7, 0.5, 0.6, 0.5, 0.7, 0.6, 0.6, 0.2, 0.2]
# Conflict graph: G[i][j] = 1 if node i and node j interfere; 0 otherwise.
G = [
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
]

T = 2           # Transmission duration in slots
sigma = 1       # Idle slot duration
num_nodes = len(transmission_prob)

# Function to convert an integer to a state vector (base-T representation)
def convert_num_to_state(n, T, num):
    return [(num // T**i) % T for i in range(n)]

# Function to convert a state vector back to its integer representation
def convert_state_to_num(n, T, state):
    total = 0
    multiplier = 1
    for j in range(n):
        total += multiplier * state[j]
        multiplier *= T
    return total

# Function to convert an integer to a binary transmission vector (length n)
def convert_num_to_trans(n, num):
    return [(num >> i) & 1 for i in range(n)]

# Given current state 'st', compute:
#   - tr_probs: row of the transition matrix
#   - tot_reward: per-node reward for successful transmission in this transition
def transition_from_state(n, T, G, probs, st):
    num_states = T ** n
    num_trans = 2 ** n
    tr_probs = [0] * num_states
    tot_reward = [0] * n

    # Iterate over all possible binary transmission decisions
    for i in range(num_trans):
        trans = convert_num_to_trans(n, i)  # binary decision vector
        valid = True
        new_state = [0] * n
        pr = 1
        for j in range(n):
            if trans[j] == 1:
                if st[j] > 0:  # Node cannot transmit if busy.
                    valid = False
                    break
                # Check interfering neighbors
                for k in range(n):
                    if G[j][k] == 1 and st[k] > 0:
                        valid = False
                if not valid:
                    break
                pr *= probs[j]  # Multiply by transmission probability.
            else:
                if st[j] == 0:
                    # If node is idle and all its neighbors are idle,
                    # factor in the probability of not transmitting.
                    for k in range(n):
                        if G[j][k] == 1 and st[k] > 0:
                            break
                    else:
                        pr *= (1 - probs[j])
        if valid:
            # Update new state and record reward
            for j in range(n):
                if trans[j] == 1:
                    new_state[j] = T - 1
                    # If no interfering neighbor transmits simultaneously, add reward.
                    for k in range(n):
                        if G[j][k] == 1 and trans[k] == 1:
                            break
                    else:
                        tot_reward[j] += pr
                elif st[j] > 0:
                    new_state[j] = st[j] - 1
            index = convert_state_to_num(n, T, new_state)
            tr_probs[index] += pr

    return tr_probs, tot_reward

# Main function: builds transition matrix, computes stationary distribution, and calculates throughput.
def markov_throughput(transmission_prob, T, sigma, G):
    num_states = T ** num_nodes
    tr_mat = np.zeros((num_states, num_states))
    # Build the transition matrix
    for s in range(num_states):
        st = convert_num_to_state(num_nodes, T, s)
        tr_probs, tot_reward = transition_from_state(num_nodes, T, G, transmission_prob, st)
        tr_mat[s, :] = tr_probs[:]
    
    # Compute stationary distribution using eigenvector of P^T with eigenvalue 1.
    evals, evecs = np.linalg.eig(tr_mat.T)
    # Select the eigenvector corresponding to eigenvalue close to 1.
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = evec1[:, 0]
    stationary = evec1 / np.sum(evec1)
    stationary = stationary.real

    # Compute throughput for each node over all states.
    throughputs = [0] * num_nodes
    for s in range(num_states):
        st = convert_num_to_state(num_nodes, T, s)
        tr_probs, tot_reward = transition_from_state(num_nodes, T, G, transmission_prob, st)
        for i in range(num_nodes):
            throughputs[i] += tot_reward[i] * stationary[s] * T

    return throughputs

# Compute and print the throughputs.
throughputs = markov_throughput(transmission_prob, T=T, sigma=sigma, G=G)
for i in range(num_nodes):
    print(f"Node {i}: Markov Throughput: {throughputs[i]:.4f}")
