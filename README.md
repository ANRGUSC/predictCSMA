
# CSMA Network Simulation

## Overview
This code simulates a Carrier Sense Multiple Access (CSMA) network protocol, featuring multiple nodes with varying transmission probabilities and interference.

## Parameters

- **Transmission probabilities**: `[0.2, 0.3, 0.5, 0.3]`
- **Adjacency matrix**: 
  ```
  [
      [0, 1, 0, 1],
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [1, 0, 1, 0]
  ]
  ```
- **Total time slots**: `10000`
- **Busy duration (T)**: `3`
- **Idle duration (sigma)**: `1`

## Functions

### `check_for_collision(busy_nodes)`

Checks for collisions among busy nodes based on the adjacency matrix.

### `simulate_csma()`

Simulates the CSMA protocol, updating node states and counting successful transmissions.

### `theoretical_saturation_throughput(transmission_prob, T, sigma, G)`

Calculates the theoretical saturation throughput for each node.

## Usage

1. **Set Parameters**: Define the transmission probabilities, adjacency matrix, total time slots, busy duration, and idle duration.
2. **Run Simulation**: Call the `simulate_csma()` function.
3. **Calculate Theoretical Throughput**: Call `theoretical_saturation_throughput()`.
4. **Print Results**: Output the success counters, saturation throughput, and theoretical saturation throughput for each node.

## Example

```python
import numpy as np

# Parameters
transmission_prob = [0.2, 0.3, 0.5, 0.3]
G = [
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
]
total_time_slots = 10000
T = 3
sigma = 1
num_nodes = len(transmission_prob)
node_states = np.zeros(num_nodes, dtype=int)
time_slot_counter = np.zeros(num_nodes, dtype=int)
success_counter = np.zeros(num_nodes, dtype=int)
state_history = []

def check_for_collision(busy_nodes):
    for node in busy_nodes:
        for neighbor in range(num_nodes):
            if G[node][neighbor] == 1 and node_states[neighbor] == 1:
                return True
    return False

def simulate_csma():
    global node_states, time_slot_counter, success_counter
    for t in range(total_time_slots):
        for i in range(num_nodes):
            if time_slot_counter[i] > 0:
                time_slot_counter[i] -= 1
                if time_slot_counter[i] == 0:
                    node_states[i] = 0

        if np.all(node_states == 0):
            for i in range(num_nodes):
                if np.random.rand() < transmission_prob[i]:
                    node_states[i] = 1
                    time_slot_counter[i] = T

            busy_nodes = np.where(node_states == 1)[0]
            if check_for_collision(busy_nodes):
                for i in busy_nodes:
                    time_slot_counter[i] = T
            elif len(busy_nodes) > 0:
                for busy_node in busy_nodes:
                    success_counter[busy_node] += 1
            for i in range(num_nodes):
                if i not in busy_nodes:
                    node_states[i] = 0
                    time_slot_counter[i] = sigma

        state_history.append(node_states.copy())

def theoretical_saturation_throughput(transmission_prob, T, sigma, G):
    num_nodes = len(transmission_prob)
    S = np.zeros(num_nodes)
    for i in range(num_nodes):
        p_i = transmission_prob[i]
        prod_other = np.prod([1 - transmission_prob[j] for j in range(num_nodes) if G[i][j] == 1 and j != i])
        numerator = p_i * prod_other * T
        denominator = sigma * prod_other * (1 - p_i) + (p_i * prod_other) * T + (1 - ((prod_other * (1 - p_i)) + p_i * prod_other)) * T
        S[i] = numerator / denominator
    return S.tolist()

simulate_csma()
theoretical_saturation_throughput = theoretical_saturation_throughput(transmission_prob, T, sigma, G)

print("Final Success Counters:")
for i in range(num_nodes):
    saturation_throughput = success_counter[i] * T / total_time_slots
    print(f"Node {i}: {success_counter[i]} successful transmissions, saturation throughput: {saturation_throughput:.4f}, theoretical saturation throughput: {theoretical_saturation_throughput[i]:.4f}")
```

## Output

The script will print the number of successful transmissions, saturation throughput, and theoretical saturation throughput for each node. Example output:

```
Final Success Counters:
Node 0: X successful transmissions, saturation throughput: Y.YYYY, theoretical saturation throughput: Z.ZZZZ
Node 1: X successful transmissions, saturation throughput: Y.YYYY, theoretical saturation throughput: Z.ZZZZ
Node 2: X successful transmissions, saturation throughput: Y.YYYY, theoretical saturation throughput: Z.ZZZZ
Node 3: X successful transmissions, saturation throughput: Y.YYYY, theoretical saturation throughput: Z.ZZZZ
```
