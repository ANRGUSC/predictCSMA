import networkx as nx
import numpy as np
import pandas as pd

# -------------------------------
# Simulation Parameters
# -------------------------------
num_nodes = 6       # Number of nodes per network
num_graphs = 5000  # Number of different random topologies to simulate
prob_edge = 0.5          # Edge creation probability for random graphs
total_time_slots = 1000000  # Total number of time slots per simulation run
T = 8              # Busy duration (time slots a node stays busy when transmitting)
sigma = 1                # zIdle duration after a collision

print(T,num_nodes)
# -------------------------------
# Function: Generate Random Graph and Transmission ProbabilitiesÅ
# -------------------------------
def generate_random_graph_and_prob(num_nodes, prob_edge):
    G = nx.erdos_renyi_graph(num_nodes, prob_edge)
    # Convert graph to adjacency matrix (as a list of lists)
    G_matrix = nx.to_numpy_array(G, dtype=int).tolist()
    # Assign a random transmission probability to each node
    transmission_prob = np.random.rand(num_nodes).tolist()
    return G_matrix, transmission_prob

# -------------------------------
# Collision Checking Functions
# -------------------------------
def check_for_collision1(node, prev_busy_slot_counter, G, num_nodes):
    # Check if any neighbor was busy in the previous time slot
    for neighbor in range(num_nodes):
        if G[node][neighbor] == 1 and prev_busy_slot_counter[neighbor] != 0:
            return True
    return False

def check_for_collision2(node, busy_slot_counter, G, num_nodes, T):
    # Check if any neighbor is just starting transmission (busy counter equals T)
    for neighbor in range(num_nodes):
        if G[node][neighbor] == 1 and busy_slot_counter[neighbor] == T:
            return True
    return False

# -------------------------------
# CSMA Simulation Function
# -------------------------------
def simulate_csma(transmission_prob, G, num_nodes, total_time_slots, T, sigma):
    # Initialize state arrays for this simulation run
    node_states = np.zeros(num_nodes, dtype=int)      # 0: idle, 1: busy
    busy_slot_counter = np.zeros(num_nodes, dtype=int)
    idle_slot_counter = np.zeros(num_nodes, dtype=int)
    success_counter = np.zeros(num_nodes, dtype=int)   # Count of successful transmissions per node

    for t in range(total_time_slots):
        # Copy busy counter from the previous time slot for collision checks
        prev_busy_slot_counter = busy_slot_counter.copy()
        
        if t == 0:
            # At time slot 0, each node may attempt to transmit
            for i in range(num_nodes):
                if np.random.rand() < transmission_prob[i]:
                    node_states[i] = 1
                    busy_slot_counter[i] = T
            busy_nodes = np.where(node_states == 1)[0]
            for node in busy_nodes:
                if not check_for_collision2(node, busy_slot_counter, G, num_nodes, T):
                    success_counter[node] += 1
        else:
            # Update counters (ensuring they don't drop below zero)
            busy_slot_counter = np.maximum(busy_slot_counter - 1, 0)
            idle_slot_counter = np.maximum(idle_slot_counter - 1, 0)
            prev_busy_slot_counter = busy_slot_counter.copy()
            
            # For each idle node (not busy and not in backoff), decide whether to transmit
            for i in range(num_nodes):
                if busy_slot_counter[i] == 0 and idle_slot_counter[i] == 0:
                    if np.random.rand() < transmission_prob[i]:
                        if not check_for_collision1(i, prev_busy_slot_counter, G, num_nodes):
                            node_states[i] = 1
                            busy_slot_counter[i] = T
                        else:
                            idle_slot_counter[i] = sigma
                            
            # For nodes that have just started transmitting (busy counter equals T),
            # check for collisions among simultaneous transmissions.
            new_busy_nodes = np.where(busy_slot_counter == T)[0]
            for node in new_busy_nodes:
                if not check_for_collision2(node, busy_slot_counter, G, num_nodes, T):
                    success_counter[node] += 1

    # Compute saturation throughput for each node
    saturation_throughput = (success_counter * T) / total_time_slots
    return saturation_throughput

# -------------------------------
# Data Generation: Run Simulations Over Multiple Topologies
# -------------------------------
all_data = []

for sim in range(num_graphs):
    G_matrix, transmission_prob = generate_random_graph_and_prob(num_nodes, prob_edge)
    throughput = simulate_csma(transmission_prob, G_matrix, num_nodes, total_time_slots, T, sigma)
    
    # Save one row per simulation run with three fields:
    # 'adj_matrix', 'transmission_prob', and 'saturation_throughput'
    result = {
        'adj_matrix': G_matrix,
        'transmission_prob': transmission_prob,
        'saturation_throughput': throughput.tolist()
    }
    all_data.append(result)
    
    if (sim + 1) % 10 == 0:
        print(f"Completed simulation {sim + 1}/{num_graphs}")

# -------------------------------
# Save Results to CSV
# -------------------------------
results_df = pd.DataFrame(all_data)
csv_filename = f"{T}_{num_nodes}_node_simulation_data_million.csv"
results_df.to_csv(csv_filename, index=False)
print(f"Data generation complete. Results saved to '{csv_filename}'")
