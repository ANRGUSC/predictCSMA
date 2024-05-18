import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# Parameters
num_samples = 1000
total_time_slots = 10000
T = 3
sigma = 1

# Function to simulate CSMA and return features and target
def simulate_csma_and_collect_data(transmission_prob, G, total_time_slots, T, sigma):
    num_nodes = len(transmission_prob)
    node_states = np.zeros(num_nodes, dtype=int)
    time_slot_counter = np.zeros(num_nodes, dtype=int)
    success_counter = np.zeros(num_nodes, dtype=int)
    
    def check_for_collision(busy_nodes):
        for node in busy_nodes:
            for neighbor in range(num_nodes):
                if G[node][neighbor] == 1 and node_states[neighbor] == 1:
                    return True
        return False

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
                    node_states[busy_node] = 1
                    time_slot_counter[busy_node] = T
                    success_counter[busy_node] += 1

    saturation_throughput = success_counter * T / total_time_slots
    avg_saturation_throughput = np.mean(saturation_throughput)
    return transmission_prob, G, saturation_throughput, avg_saturation_throughput

# Generate dataset
dataset = []
avg_throughputs = []

for _ in range(num_samples):
    transmission_prob = np.random.rand(4)  # Example for 4 nodes
    G = np.random.randint(0, 2, (4, 4))
    np.fill_diagonal(G, 0)
    features, graph, target, avg_throughput = simulate_csma_and_collect_data(transmission_prob, G, total_time_slots, T, sigma)
    edge_index = np.array(np.nonzero(graph))
    data = Data(x=torch.tensor(features, dtype=torch.float).view(-1, 1), 
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(target, dtype=torch.float).view(1, -1))  # Modify target shape
    dataset.append(data)
    avg_throughputs.append(avg_throughput)


# Split into training and testing sets
train_dataset = dataset[:int(0.8 * len(dataset))]
test_dataset = dataset[int(0.8 * len(dataset)):]

# Define the GNN model with GAT layer
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.gat2 = GATConv(8 * 8, 16, heads=8, concat=False, dropout=0.6)
        self.fc = torch.nn.Linear(16, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x

# Hyperparameters
in_channels = 1
out_channels = 4
model = GATNet(in_channels, out_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.MSELoss()

# DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(out.shape))  # Ensure target shape matches output shape
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            error += criterion(out, data.y.view(out.shape)).item() 
    return error / len(loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(2000):
    train()
    train_error = test(train_loader)
    test_error = test(test_loader)
    if epoch % 200 == 0:
      print(f'Epoch: {epoch}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}')

# Evaluate model performance
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        data = data.to(device)
        out = model(data)
        actual = data.y.cpu().numpy()
        prediction = out.cpu().numpy()
        print(f"Test Sample {i+1}:")
        print(f"Actual Saturation Throughput: {np.mean(actual)}")
        print(f"Predicted Saturation Throughput: {np.mean(prediction)}\n")
