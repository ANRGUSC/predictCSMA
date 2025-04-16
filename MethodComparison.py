import numpy as np
import pandas as pd
import networkx as nx

# Simulate CSMA
def simulate_csma(G, transmission_prob, T, sigma, total_time_slots):
    num_nodes = len(transmission_prob)
    node_states, busy_slot_counter, idle_slot_counter, success_counter = reinitialize_state_arrays(num_nodes)
    
    for t in range(total_time_slots):
        prev_busy_slot_counter = [busy_slot_counter[i] for i in range(num_nodes)]
        if t == 0:
            for i in range(num_nodes):
                if np.random.rand() < transmission_prob[i]:
                    node_states[i] = 1
                    busy_slot_counter[i] = T
            busy_nodes = np.where(node_states == 1)[0]
            if len(busy_nodes) > 0:
                for node in busy_nodes:
                    if check_for_collision2(node, G, T, busy_slot_counter):
                        pass
                    else:
                        success_counter[node] += 1
        else:
            for i in range(num_nodes):
                if busy_slot_counter[i] > 0:
                    busy_slot_counter[i] -= 1
                if idle_slot_counter[i] > 0:
                    idle_slot_counter[i] -= 1
            prev_busy_slot_counter = [busy_slot_counter[i] for i in range(num_nodes)]
            for i in range(num_nodes):
                if prev_busy_slot_counter[i] == 0 and idle_slot_counter[i] == 0:
                    if np.random.rand() < transmission_prob[i]:
                        if not check_for_collision1(i, G, prev_busy_slot_counter):
                            node_states[i] = 1
                            busy_slot_counter[i] = T
                        else:
                            idle_slot_counter[i] = sigma
            busy_nodes = np.where(busy_slot_counter != 0)[0]
            new_busy_nodes = np.where(busy_slot_counter == T)[0]
            if len(new_busy_nodes) > 0:
                for node in new_busy_nodes:
                    if check_for_collision2(node, G, T, busy_slot_counter):
                        pass
                    else:
                        success_counter[node] += 1
    return np.mean(success_counter * T / total_time_slots)

# Collision check functions
def check_for_collision1(node, G, prev_busy_slot_counter):
    for neighbor in range(len(G)):
        if G[node][neighbor] == 1 and prev_busy_slot_counter[neighbor] != 0:
            return True
    return False

def check_for_collision2(node, G, T, busy_slot_counter):
    for neighbor in range(len(G)):
        if G[node][neighbor] == 1 and busy_slot_counter[neighbor] == T:
            return True
    return False

# Theoretical saturation throughput Approximation
def theoretical_saturation_throughput_old(transmission_prob, T, sigma, G):
    num_nodes = len(transmission_prob)
    S = np.zeros(num_nodes)
    for i in range(num_nodes):
        p_i = transmission_prob[i]
        prod_other = np.prod([1 - transmission_prob[j] for j in range(num_nodes) if G[i][j] == 1 and j != i])
        numerator = p_i * prod_other * T
        denominator = sigma * prod_other * (1 - p_i) + (p_i * prod_other) * T + (1 - ((prod_other * (1 - p_i)) + p_i * prod_other)) * T
        S[i] = numerator / denominator
    return np.mean(S)  # Return the average throughput

#Renewal Classic
def theoretical_saturation_throughput_oldest(transmission_prob, T, sigma, G):
    num_nodes = len(transmission_prob)
    S = np.zeros(num_nodes)
    for i in range(num_nodes):
        p_i = transmission_prob[i]
        prod_other = np.prod([1 - transmission_prob[j] for j in range(num_nodes) if j != i])
        numerator = p_i * prod_other * T
        denominator = sigma * prod_other * (1 - p_i) + (p_i * prod_other) * T + (1 - ((prod_other * (1 - p_i)) + p_i * prod_other)) * T
        S[i] = numerator / denominator
    return np.mean(S)  # Return the average throughput

# Markov theoretical method
def convert_num_to_state(n,T,num):
  st = [0 for i in range(n)]
  for i in range(n):
    st[i] = num%T
    num = num//T
    # print(st)
  return st

def convert_state_to_num(n,T,state):
   sum = 0
   a = 1
   for j in range(n):
        sum = sum+ a*state[j]
        a *=T
#    print(sum)
   return sum

def convert_num_to_trans(n, num):
  st = [0 for i in range(n)]
  for i in range(n):
    st[i] = num%2
    num = num//2
  return st

def transition_from_state(n,T,G,probs,st):
  len_st_space = T**n
  a = 2**n
  tr_probs = [0 for i in range(len_st_space)]
  tot_reward = [0 for i in range(n)]
  for i in range(a):
    trans = convert_num_to_trans(n, i)
    valid = 1
    state = [0 for j in range(n)]
    pr = 1
    for j in range(n):
       if(trans[j] == 1):
         if(st[j] > 0):
             valid = 0
             break
         for k in range(n):
            if(G[j][k] == 1 and st[k] > 0):
               valid = 0
         if(valid == 0):
            break
         else:
            pr*=probs[j]
       else:
         if(st[j] == 0):
           for k in range(n):
            if(G[j][k] == 1 and st[k] > 0):
               break
           else:
               pr*=(1-probs[j])
    if(valid == 1):
        for j in range(n):
           if(trans[j] == 1):
             state[j] = T-1
             for k in range(n):
               if(G[j][k] == 1 and trans[k] == 1):
                  break
             else:
                tot_reward[j] += pr
           elif(st[j]>0):
             state[j] = st[j] -1
        st_number =  convert_state_to_num(n,T,state)
        tr_probs[st_number] += pr
  return tr_probs, tot_reward

def dfs(n,st_probs):
  comps = [0]
  visited = [0 for i in range(n)]
  visited[0] = 1
  for k in range(n):
    if(len(comps) == 0):
       return visited
    current = comps.pop(0)
    visited[current] = 1
    for j in range(n):
      if(st_probs[current][j] > 0 and visited[j] == 0):
          comps.append(j)
  return visited

def theoretical_saturation_throughput_new(transmission_prob, T, sigma, G):
    num_states = T**num_nodes
    tr_mat = np.zeros((num_states,num_states))
    for k in range(num_states):
        st = convert_num_to_state(num_nodes,T,k)
        tr_probs, tot_reward =  transition_from_state(num_nodes,T,G,transmission_prob,st)
        for j in range(num_states):
            tr_mat[k][j] = tr_probs[j]
    visited = dfs(num_states,tr_mat)
    m_tr_mat = []
    for k in range(num_states):
        if(visited[k] == 1):
            aa = []
            for j in range(num_states):
                if(visited[j] == 1):
                    aa.append(tr_mat[k][j])
            m_tr_mat.append(aa)
    m_tr_mat = np.array(m_tr_mat)
    #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(tr_mat.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real

    throughputs = [0 for i in range(num_nodes)]
    count = 0
    for j in range(num_states):
            st = convert_num_to_state(num_nodes,T,j)
            tr_probs, tot_reward =  transition_from_state(num_nodes,T,G,transmission_prob,st)
            print(len(tot_reward),len(throughputs),len(stationary))
            for i in range(num_nodes):
                throughputs[i] += tot_reward[i]*stationary[j]*T
            # count+=1
    return throughputs

# Generate Erdős-Rényi graph
def generate_erdos_renyi_graph(num_nodes, edge_prob):
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    adjacency_matrix = nx.to_numpy_array(G)
    return adjacency_matrix

# Reinitialize state arrays
def reinitialize_state_arrays(num_nodes):
    return np.zeros(num_nodes, dtype=int), np.zeros(num_nodes, dtype=int), np.zeros(num_nodes, dtype=int), np.zeros(num_nodes, dtype=int)

# Parameters
# edge_probs = [0.1,0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8, 0.9,1]  # Different edge densities
edge_probs = [0.1, 0.3,0.5, 0.7, 0.9,1]  # Different edge densities

num_nodes = 10  # Number of nodes in the graph
T = 3
sigma = 1
total_time_slots = 1000000
num_graphs_per_density = 1  # Number of graphs to generate for each edge density
results = []

# Simulation and comparison
for edge_prob in edge_probs:
    sim_throughputs = []
    theoretical_throughputs_old = []
    theoretical_throughputs_new = []
    theoretical_throughputs_oldest = []
    
    for _ in range(num_graphs_per_density):
        G = generate_erdos_renyi_graph(num_nodes, edge_prob)
        transmission_prob = np.random.rand(num_nodes)
        avg_simulation_throughput = simulate_csma(G, transmission_prob, T, sigma, total_time_slots)
        avg_theoretical_throughput_old = theoretical_saturation_throughput_old(transmission_prob, T, sigma, G)
        avg_theoretical_throughput_oldest = theoretical_saturation_throughput_oldest(transmission_prob, T, sigma, G)
        avg_theoretical_throughput_new = theoretical_saturation_throughput_new(transmission_prob, T, sigma, G)
        
        sim_throughputs.append(avg_simulation_throughput)
        theoretical_throughputs_old.append(avg_theoretical_throughput_old)
        theoretical_throughputs_oldest.append(avg_theoretical_throughput_oldest)
        theoretical_throughputs_new.append(avg_theoretical_throughput_new)
    
    # Calculate averages across all generated graphs
    avg_simulation_throughput = np.mean(sim_throughputs)
    avg_theoretical_throughput_old = np.mean(theoretical_throughputs_old)
    avg_theoretical_throughput_oldest = np.mean(theoretical_throughputs_oldest)
    avg_theoretical_throughput_new = np.mean(theoretical_throughputs_new)
    
    results.append({
        "Edge Probability": edge_prob,
        "Average Simulation Throughput": avg_simulation_throughput,
        "Average Renewal Theory (approximation)": avg_theoretical_throughput_old,
        "Average Renewal Theory (classic)": avg_theoretical_throughput_oldest,
        "Average Markov Theoretical": avg_theoretical_throughput_new,
        "Difference (approximation - Simulation)": avg_theoretical_throughput_old - avg_simulation_throughput,
        "Difference (classic - Simulation)": avg_theoretical_throughput_oldest - avg_simulation_throughput,
        "Difference (Markov - Simulation)": avg_theoretical_throughput_new - avg_simulation_throughput,
        "Difference (Markov - Old)": avg_theoretical_throughput_new - avg_theoretical_throughput_old,
    })

# Creating the table
comparison_table = pd.DataFrame(results)

# Save the table as a CSV file
csv_path = 'comparison_table.csv'
comparison_table.to_csv(csv_path, index=False)

print(f"Comparison table saved as {csv_path}")
