import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the data from the pickle file
filename = 'data/data_10000_samples_4_nodes.pkl'  # Update this path to your actual file location

with open(filename, 'rb') as f:
    dataset = pickle.load(f)

# Extract transmission probabilities and rates for each node
transmission_probs = []
rates = []

for data in dataset[:1000]:  # Limit to 1000 samples
    node_probs = data['x'].numpy().flatten()  # Extract transmission probabilities for all nodes
    target_rates = data['y'].numpy().flatten()  # Extract target rates for all nodes
    
    transmission_probs.append(node_probs)
    rates.append(target_rates)

# Convert lists to numpy arrays
transmission_probs = np.array(transmission_probs)
rates = np.array(rates)

# Ensure both arrays have the same length
assert len(transmission_probs) == len(rates), "Mismatch in the number of samples between transmission_probs and rates"

# Define the theoretical saturation throughput function
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

# Define network parameters
T =3  # Example value, adjust based on your network
sigma = 1  # Example value, adjust based on your network
G = np.array([[0, 1, 1, 1],  # Example adjacency matrix, adjust based on your network
              [1, 0, 1, 1],
              [1, 1, 0, 1],
              [1, 1, 1, 0]])

# Calculate theoretical saturation throughput for each sample
theoretical_throughputs = [theoretical_saturation_throughput(probs, T, sigma, G) for probs in transmission_probs]

# Define the neural network model
class RatePredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RatePredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Hyperparameters
input_size = 4  # Number of nodes
hidden_size = 64
output_size = 4  # Predicting rates for all nodes
num_epochs = 100
learning_rate = 0.001

# Prepare data
X = transmission_probs
y = rates

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Initialize the model, loss function, and optimizer
model = RatePredictionNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            val_losses.append(val_loss.item())
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = mean_squared_error(y_test.numpy(), predictions.numpy())
    test_mae = mean_absolute_error(y_test.numpy(), predictions.numpy())
    test_r2 = r2_score(y_test.numpy(), predictions.numpy())
    print(f'Test Loss (MSE): {test_loss:.4f}')
    print(f'Test Mean Absolute Error (MAE): {test_mae:.4f}')
    print(f'Test R-squared (R2): {test_r2:.4f}')

# Plot predictions vs actual values for each node
for node_index in range(output_size):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions[:, node_index], label='Predictions', linestyle='--')
    plt.plot(y_test[:, node_index], label='Simulation', alpha=0.7)
    theoretical = [theoretical_saturation_throughput(X_test[i].numpy(), T, sigma, G)[node_index] for i in range(len(X_test))]
    plt.plot(theoretical, label='Theoretical', linestyle=':')
    plt.xlabel('Sample Index')
    plt.ylabel('Rate')
    plt.legend()
    plt.title(f'Node {node_index + 1} - Predictions vs Actual Values vs Theoretical Values')
    plt.show()

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
if len(val_losses) > 0:
    plt.plot(np.arange(10, num_epochs + 1, 10), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
