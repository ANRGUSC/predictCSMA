

import numpy as np
transmission_prob=[0.2, 0.3, 0.7,0.3,0.6,0.5,0.7,0.6,0.6,0.2]
G= ([
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    
# Number of nodes
num_nodes = len(transmission_prob)

# Define T (busy duration) and sigma (idle duration)
T = 2
sigma = 1

def theoretical_saturation_throughput_old(transmission_prob, T, sigma, G):
    num_nodes = len(transmission_prob)
    S = np.zeros(num_nodes)
    for i in range(num_nodes):
        p_i = transmission_prob[i]
        prod_other = np.prod([1 - transmission_prob[j] for j in range(num_nodes) if G[i][j] == 1 and j != i])
        numerator = p_i * prod_other * (T)
        denominator = sigma * prod_other * (1 - p_i) + (p_i * prod_other) * T + (1- ((prod_other * (1 - p_i)) + p_i * prod_other)) * T
        S[i] = numerator / denominator
    return S.tolist()
   

theoretical_saturation_throughput = theoretical_saturation_throughput_old(transmission_prob, T=T, sigma=sigma, G=G)
print("T=",T)
for i in range(num_nodes):
    print(f"Node {i}: Renewal_Throughput: {theoretical_saturation_throughput[i]:.4f}")
