import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, chain
import pandas as pd


transmission_prob = [0.2, 0.1, 0.7, 0.5, 0.6, 0.5, 0.7, 0.6, 0.6, 0.2, 0.2]
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

T = 2
sigma = 1

total_time_slots = 1000000
# Number of nodes
num_nodes = len(transmission_prob)


# Initialize state arrays
node_states = np.zeros(num_nodes, dtype=int)  # 0: idle, 1: busy
busy_slot_counter = np.zeros(num_nodes, dtype=int)
idle_slot_counter = np.zeros(num_nodes, dtype=int)
success_counter = np.zeros(num_nodes, dtype=int)  # Counter for successful transmissions


def check_for_collision1(node):
    for neighbor in range(num_nodes):
        if G[node][neighbor] == 1:
             if prev_busy_slot_counter[neighbor] != 0:
                return True
    return False

def check_for_collision2(node):
    for neighbor in range(num_nodes):
        if G[node][neighbor] == 1:
             if busy_slot_counter[neighbor] == T:
                return True
    return False

def simulate_csma():
    global node_states, busy_slot_counter, success_counter, prev_busy_slot_counter
    for t in range(total_time_slots):
        print(f"Time Slot {t + 1}")
        prev_busy_slot_counter = [busy_slot_counter[i] for i in range(num_nodes)]
        if t ==0:
            for i in range(num_nodes):
                    if np.random.rand() < transmission_prob[i]:
                        print("node ",i,"become busy")
                        node_states[i] = 1
                        busy_slot_counter[i] = T
            busy_nodes = np.where(node_states == 1)[0]
            idle_nodes = np.where(node_states == 0)[0]
            print("idel nodes: ",idle_nodes)
            print("Busy nodes: ",busy_nodes)
            if(len(busy_nodes)>0):
                for node in busy_nodes:
                    if check_for_collision2(node):
                        print(f"Node {node} had collision for {busy_slot_counter[node]}")
                    else:
                        success_counter[node] += 1  # Increment success counter
                        print(f"Nodes {node} successfully transmitting for {busy_slot_counter[node]}")
            else:
                print(f"No transmission occurred")
            
                        
        else:
        # Update states based on counters
            for i in range(num_nodes):
                if busy_slot_counter[i] > 0 :
                    busy_slot_counter[i] -= 1
                if idle_slot_counter[i] > 0 :
                    idle_slot_counter[i] -= 1
            prev_busy_slot_counter = [busy_slot_counter[i] for i in range(num_nodes)]
            for i in range(num_nodes):
                if prev_busy_slot_counter[i] == 0 and idle_slot_counter[i]==0:
                    if np.random.rand() < transmission_prob[i]:
                        print("node ",i,"wants to transmit")
                        if not check_for_collision1(i):
                            print("node ",i,"become busy")
                            node_states[i] = 1
                            busy_slot_counter[i] = T
                        else:
                            print("neighbors of ",i,"are transmitting it can not transmit")
                            idle_slot_counter[i] = sigma
                    # Check for collisions among neighbors
            busy_nodes = np.where(busy_slot_counter != 0)[0]
            new_busy_nodes = np.where(busy_slot_counter == T)[0]
            print("Busy nodes: ",busy_nodes)
            print("New Busy nodes: ",new_busy_nodes)

            if(len(new_busy_nodes)>0):
                for node in new_busy_nodes:
                    if check_for_collision2(node):
                        print(f"Node {node} had collision for {busy_slot_counter[node]}.")
                    else:
                        success_counter[node] += 1  # Increment success counter
                        print(f"Node {node} successfully transmitting for {busy_slot_counter[node]}.")


        print(f"Node states: {node_states}")
        print(f"Time slot counters: {busy_slot_counter}")
        print(f"Idle slot counters: {idle_slot_counter}")
        print(f"Success counters: {success_counter}\n")

simulate_csma()
print("Final Success Counters:")
saturation_throughput = success_counter*T/ total_time_slots

for i in range(num_nodes):
    print(f"Node {i}: Simulation_Throughput: {saturation_throughput[i]:.4f}")
