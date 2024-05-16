# -*- coding: utf-8 -*-
import numpy as np

# Transmission probabilities
transmission_prob = [0.2, 0.3, 0.5, 0.3]

# Adjacency matrix
G = [
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
]

# Number of time slots for the simulation
total_time_slots = 10000

# Number of nodes
num_nodes = len(transmission_prob)

# Define T (busy duration) and sigma (idle duration)
T = 3
sigma = 1

# Initialize state arrays
node_states = np.zeros(num_nodes, dtype=int)  # 0: idle, 1: busy
time_slot_counter = np.zeros(num_nodes, dtype=int)
success_counter = np.zeros(num_nodes, dtype=int)  # Counter for successful transmissions


def check_for_collision(busy_nodes):
    for node in busy_nodes:
        for neighbor in range(num_nodes):
            if G[node][neighbor] == 1 and node_states[neighbor] == 1:
                return True
    return False

def simulate_csma():
    global node_s, time_slot_counter, success_counter
    for t in range(total_time_slots):
        print(f"Time Slot {t + 1}")

        # Update s based on counters
        for i in range(num_nodes):
            if time_slot_counter[i] > 0:
                time_slot_counter[i] -= 1
                if time_slot_counter[i] == 0:
                    node_s[i] = 0

        # Check if all nodes are idle
        if np.all(node_s == 0):
            # Determine new s based on transmission probabilities
            for i in range(num_nodes):
                if np.random.rand() < transmission_prob[i]:
                    node_s[i] = 1
                    time_slot_counter[i] = T

            # Check for collisions among neighbors
            busy_nodes = np.where(node_s == 1)[0]
            if check_for_collision(busy_nodes):
                for i in busy_nodes:
                    time_slot_counter[i] = T
                print(f"Collision occurred among neighbors, nodes reset to idle after {T} time slots.")
            elif len(busy_nodes) > 0:
                for busy_node in busy_nodes:
                    node_s[busy_node] = 1
                    time_slot_counter[busy_node] = T
                    success_counter[busy_node] += 1  # Increment success counter
                print(f"Nodes {busy_nodes} successfully transmitting for {T} time slots.")
            else:
                print(f"No transmission occurred, nodes wait for {sigma} time slot.")
                for i in range(num_nodes):
                    node_s[i] = 0
                    time_slot_counter[i] = sigma

        print(f"Node states: {node_states}")
        print(f"Time slot counters: {time_slot_counter}")
        print(f"Success counters: {success_counter}\n")


def therotical_saturation_throughput(transmission_prob, T, sigma, G):

    num_nodes = len(transmission_prob)
    S = np.zeros(num_nodes)

    for i in range(num_nodes):
        p_i = transmission_prob[i]
        prod_other = np.prod([1 - transmission_prob[j] for j in range(num_nodes) if G[i][j] == 1 and j != i])
        numerator = p_i * prod_other * T
        denominator = sigma * prod_other * (1 - p_i) + (p_i * prod_other) * T + (1- ((prod_other * (1 - p_i)) + p_i * prod_other)) * T

        S[i] = numerator / denominator

    return S.tolist()

# Run the CSMA simulation
simulate_csma()
theoretical_saturation_throughput = therotical_saturation_throughput(transmission_prob, T=T, sigma=sigma, G=G)

# Print the final success counters
print("Final Success Counters:")
for i in range(num_nodes):
    saturation_throughput = success_counter*T / total_time_slots
    print(f"Node {i}: {success_counter[i]} successful transmissions, saturation throughput: {saturation_throughput[i]:.4f}, theoretical saturation throughput: {theoretical_saturation_throughput[i]:.4f}")
