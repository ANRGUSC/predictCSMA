
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv,GINConv, SAGEConv
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
  
class GNN_EdgeAttr(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GNN_EdgeAttr, self).__init__()
        self.convs = nn.ModuleList()
        # self.dropout = dropout

        # Helper function to build an MLP for the GINEConv layer.
        def build_mlp(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )

        # Input layer: use edge_dim=1 for 1D edge attributes.
        self.convs.append(GINEConv(build_mlp(input_dim, hidden_dims[0]), edge_dim=1))
        # self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

        # Hidden layers.
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GINEConv(build_mlp(hidden_dims[i], hidden_dims[i+1]), edge_dim=1))

        # Additional Graph layer (optional).
        self.convs.append(GINEConv(build_mlp(hidden_dims[-1], hidden_dims[-1]), edge_dim=1))

        # Fully Connected Layers.
        self.fc1 = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2)
        self.fc2 = nn.Linear(hidden_dims[-1] // 2, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        # Fully connected layers.
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)  # Ensure outputs are in [0,1]
        return x

class GCNwithMlp(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        self.convs = nn.ModuleList()
        self.mlps  = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        # Add GCNConv + MLP for each layer
        for i in range(len(hidden_dims)):
            self.convs.append(GCNConv(dims[i], dims[i+1]))
            self.mlps.append(nn.Sequential(
                nn.Linear(dims[i+1], dims[i+1]),
                nn.ReLU(),
                nn.Linear(dims[i+1], dims[i+1]),
            ))

        # Extra GCNConv + MLP layer (parity with your GINNet)
        self.convs.append(GCNConv(hidden_dims[-1], hidden_dims[-1]))
        self.mlps.append(nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
        ))

        # MLP head
        self.fc1 = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2)
        self.fc2 = nn.Linear(hidden_dims[-1] // 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, mlp in zip(self.convs, self.mlps):
            x = conv(x, edge_index)
            x = mlp(x)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

class SAGEwithMlp(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        self.convs = nn.ModuleList()
        self.mlps  = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            self.convs.append(SAGEConv(dims[i], dims[i+1],aggr='add'))
            self.mlps.append(nn.Sequential(
                nn.Linear(dims[i+1], dims[i+1]),
                nn.ReLU(),
                nn.Linear(dims[i+1], dims[i+1]),
            ))

        # Extra GCNConv + MLP layer (parity with your GINNet)
        self.convs.append(SAGEConv(hidden_dims[-1], hidden_dims[-1],aggr='add'))
        self.mlps.append(nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
        ))

        # MLP head
        self.fc1 = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2)
        self.fc2 = nn.Linear(hidden_dims[-1] // 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, mlp in zip(self.convs, self.mlps):
            x = conv(x, edge_index)
            x = mlp(x)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


class GINNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        self.convs = nn.ModuleList()

        # helper: internal MLP for each GINConv
        def gin_mlp(in_c, out_c):
            return nn.Sequential(
                nn.Linear(in_c,  out_c),
                nn.ReLU(),
                nn.Linear(out_c, out_c),
            )

        # ─ first graph layer
        self.convs.append(GINConv(gin_mlp(input_dim, hidden_dims[0]),train_eps=True))

        # ─ hidden graph layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GINConv(gin_mlp(hidden_dims[i],
                                              hidden_dims[i + 1]),
                                      train_eps=True))

        # ─ extra graph layer (keeps parity with other models)
        last = hidden_dims[-1]
        self.convs.append(GINConv(gin_mlp(last, last), train_eps=True))

        # ─ MLP head
        self.fc1 = nn.Linear(last, last // 2)
        self.fc2 = nn.Linear(last // 2, output_dim)

    # ------------------------------------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

       
class DGCNConv(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, mlp_layers=[64, 32]):
        super(DGCNConv, self).__init__()
        
        self.layers = nn.ModuleList()
        # self.dropout = dropout
        
        # Build the layer dimensions
        dims = [input_dim] + hidden_dims + [hidden_dims[-1]]  # Keep last hidden dim
        
        # Create  DGCN layers
        for i in range(len(dims) - 1):
            layer = DGCNLayer(dims[i], dims[i+1])
            self.layers.append(layer)
        
        # Add MLP at the end
        mlp_dims = [hidden_dims[-1]] + mlp_layers + [output_dim]
        mlp_layers_list = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers_list.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
            if i < len(mlp_dims) - 2:  # No ReLU on last layer
                mlp_layers_list.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers_list)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Pass through all GCN layers
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        # Apply MLP
        x = self.mlp(x)
        
        return torch.sigmoid(x)


class DGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.weight_neigh = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_self  = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.neighbor_importance = nn.Linear(out_channels, 1)


        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        
    # ----- helpers ---------------------------------------------------------
    def reset_parameters(self):
        glorot(self.weight_neigh)
        glorot(self.weight_self)
        zeros(self.bias)

    # ----- forward pass ----------------------------------------------------
    def forward(self, x, edge_index):
        # 1. linear transform applied to the features that will travel
        x_neigh = torch.matmul(x, self.weight_neigh)

        # 2. aggregate un-normalised neighbour messages
        out = self.propagate(edge_index, x=x_neigh, size=(x.size(0), x.size(0)))

        # 3. add self-transformed feature
        out = out + torch.matmul(x, self.weight_self)

        if self.bias is not None:
            out = out + self.bias
        return out

    # ----- MessagePassing hooks -------------------------------------------
    def message(self, x_j):
        # no scaling, just forward the neighbour’s feature
        importance = torch.sigmoid(self.neighbor_importance(x_j))
        return importance*F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out
