
"""
Utility script to load a trained **GCNConv** model and predict
per-node **saturation throughput**.

The script supports

* **one graph** supplied inline (see bottom) **or**
* **many graphs** in a **CSV** with the columns  
  ─ `adj_matrix`            – N × N 0/1 list (Python-literal)  
  ─ `transmission_prob`     – list of N floats (0 < p < 1)  
  ─ `saturation_throughput` – list of N floats (optional → MAE / NMAE)

------------------------------------------------------------------------
Quick start
-----------
1.  Put your CSV next to this file, e.g. `evaluation_set.csv`.
2.  Update `CSV_PATH` below.
3.  Run: `python predict_saturation_throughput_enhanced_gcn.py`
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

# Import your exact architecture classes
try:
    from GNN_Architecture import DGCNConv, EnhancedGCNLayerNoNorm
except ImportError:
    # Define the exact architecture here if import fails
    class EnhancedGCNLayerNoNorm(MessagePassing):
        """
        Same "self-vs-neighbour" idea as before, but
        • simply **sums** neighbour messages (aggr='add')
        • no mean / degree normalisation
        """
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
            
        def reset_parameters(self):
            glorot(self.weight_neigh)
            glorot(self.weight_self)
            zeros(self.bias)

        def forward(self, x, edge_index):
            x_neigh = torch.matmul(x, self.weight_neigh)
            out = self.propagate(edge_index, x=x_neigh, size=(x.size(0), x.size(0)))
            out = out + torch.matmul(x, self.weight_self)

            if self.bias is not None:
                out = out + self.bias
            return out

        def message(self, x_j):
            importance = torch.sigmoid(self.neighbor_importance(x_j))
            return importance * F.relu(x_j)

        def update(self, aggr_out):
            return aggr_out

    class EnhancedGCNConv(nn.Module):
        """
        Enhanced GCN that incorporates key RGCN techniques: add mlp at the e
        - Separate transformations for self and neighbor features
        - Mean aggregation like RGCN
        - No symmetric normalization
        This architecture mimics RGCN behavior for better performance
        """
        def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, mlp_layers=[64, 32]):
            super(EnhancedGCNConv, self).__init__()
            
            self.layers = nn.ModuleList()
            
            # Build the layer dimensions
            dims = [input_dim] + hidden_dims + [hidden_dims[-1]]  # Keep last hidden dim
            
            # Create enhanced GCN layers
            for i in range(len(dims) - 1):
                layer = EnhancedGCNLayerNoNorm(dims[i], dims[i+1])
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

# ---------------------------------------------------------------------------
# Configuration – match your training settings
# ---------------------------------------------------------------------------
T: int = 7                     # Your training parameter          # Sanity check for inline example
MODEL_PATH: str = "7_bestenhanced.pt" # Produced by your training script

# Model architecture - MUST match your training exactly
INPUT_DIM: int = 1              # Only p_i (transmission probability)
HIDDEN_DIMS: list[int] = [64, 64, 64, 64, 64,64,64]  # Your 7-layer architecture
OUTPUT_DIM: int = 1

# Data source
CSV_PATH: str = "5_data_million.csv"
# Leave CSV_PATH = "" to run the inline single-graph demo

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_data_object(adj_matrix: list[list[int]],
                      transmission_prob: list[float]) -> Data:
    """Create a torch_geometric Data object matching your training format."""
    tp = np.asarray(transmission_prob, dtype=float)

    # Node features: only p_i (transmission probability)
    x = torch.tensor(np.column_stack([tp]), dtype=torch.float)

    # Build edge list
    edges = [[i, j] for i, row in enumerate(adj_matrix)
             for j, val in enumerate(row) if val == 1]
    
    if not edges:
        # Handle disconnected graphs
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create data object (no edge attributes needed for your model)
    data = Data(x=x, edge_index=edge_index, 
                y=torch.zeros((len(tp), 1)))  # Dummy target for compatibility
    
    return data

def make_data(adj, p_vec, sat_thr):
    """Build torch_geometric.Data with node feature = p_i (shape n×1)."""
    edges = [[i, j] for i, row in enumerate(adj)
                     for j, v in enumerate(row) if v == 1]
    if not edges:
        return None
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    N = len(p_vec)
    

    # x = torch.tensor(np.array(p_vec).reshape(-1, 1), dtype=torch.float)
    x = torch.tensor(np.column_stack([p_vec
                                      ]), dtype=torch.float)
    y = torch.tensor((sat_thr), dtype=torch.float).view(-1, 1)

    return Data(x=x, edge_index=edge_index, y=y)


def load_trained_model(path: str) -> EnhancedGCNConv:
    """Load your trained Enhanced GCN model with exact architecture."""
    model = EnhancedGCNConv(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, 
                           dropout=0.5, mlp_layers=[64, 32])  # Match training exactly
    
    try:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        
        # Verify model loaded correctly
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model loaded: {total_params:,} parameters")
        
        # Quick sanity check
        dummy_x = torch.randn(4, INPUT_DIM)
        dummy_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index)
        
        with torch.no_grad():
            test_output = model(dummy_data)
        
        if torch.isnan(test_output).any():
            print("⚠️ Warning: Model outputs NaN values")
        elif test_output.min() < 0 or test_output.max() > 1:
            print(f"⚠️ Warning: Output range [{test_output.min():.4f}, {test_output.max():.4f}] unusual")
        else:
            print(f"✓ Model sanity check passed: output range [{test_output.min():.4f}, {test_output.max():.4f}]")
        
        return model
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


# Load the model once
print(f"Loading model from: {MODEL_PATH}")
try:
    MODEL = load_trained_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print(f"✗ Model file not found: {MODEL_PATH}")
    print("Make sure you've trained the model first and the file path is correct.")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)


def predict_saturation_throughput(adj_matrix: list[list[int]],
                                  transmission_probabilities: list[float]) -> np.ndarray:
    """Predict saturation throughput for given graph and transmission probabilities."""
    
    # Build data object exactly like training
    tp = np.asarray(transmission_probabilities, dtype=float)
    x = torch.tensor(np.column_stack([tp]), dtype=torch.float)  # Match training format exactly
    
    # Build edge list exactly like training
    edges = [[i, j] for i, row in enumerate(adj_matrix)
             for j, val in enumerate(row) if val == 1]
    
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create data object (no y needed for prediction)
    data = Data(x=x, edge_index=edge_index)
    
    # Ensure model is in eval mode
    MODEL.eval()
    
    with torch.no_grad():
        predictions = MODEL(data)  # Should output values in [0,1] due to sigmoid
    
    return predictions.squeeze(-1).cpu().numpy()


def compute_mae_metrics(pred: np.ndarray,
                        actual: np.ndarray) -> tuple[float, float]:
    """Compute Mean Absolute Error and Normalized MAE."""
    mae = np.mean(np.abs(pred - actual))
    nmae = mae / np.mean(actual) if np.mean(actual) > 0 else 0.0
    return mae, nmae


# ---------------------------------------------------------------------------
# CSV pipeline
# ---------------------------------------------------------------------------

def run_csv_evaluation(path: str) -> None:
    """Evaluate model on CSV dataset."""
    if not Path(path).exists():
        print(f"✗ CSV file not found: {path}")
        return
        
    df = pd.read_csv(Path(path))
    required_cols = {"adj_matrix", "transmission_prob"}
    
    if not required_cols.issubset(df.columns):
        print(f"✗ CSV must have columns: {required_cols}")
        print(f"Found columns: {set(df.columns)}")
        return

    all_pred, all_act = [], []
    has_ground_truth = "saturation_throughput" in df.columns

    print(f"\n{'='*80}")
    print(f"Evaluating {len(df)} graphs from '{Path(path).name}'")
    print(f"Model: Enhanced GCN with {sum(p.numel() for p in MODEL.parameters() if p.requires_grad):,} parameters")
    print(f"{'='*80}")

    for idx, row in df.iterrows():
        try:
            # Parse data
            G = ast.literal_eval(row["adj_matrix"])
            tp = ast.literal_eval(row["transmission_prob"])
            
            # Predict
            pred = predict_saturation_throughput(G, tp)
            
            print(f"\nGraph {idx+1:3d}:")
            print(f"  Nodes: {len(tp)}")
            print(f"  Edges: {sum(sum(row) for row in G)}")
            print(f"  Transmission probs: {[f'{p:.3f}' for p in tp]}")
            print(f"  Predicted throughput: {[f'{p:.6f}' for p in pred]}")
            
            all_pred.extend(pred)
            
            # Compare with ground truth if available
            if has_ground_truth and pd.notna(row["saturation_throughput"]):
                actual = ast.literal_eval(row["saturation_throughput"])
                mae, nmae = compute_mae_metrics(pred, np.array(actual))
                print(f"  Actual throughput:    {[f'{a:.6f}' for a in actual]}")
                print(f"  MAE: {mae:.6f}, NMAE: {nmae:.4f} ({nmae*100:.2f}%)")
                all_act.extend(actual)
            
            print("-" * 80)
            
        except Exception as e:
            print(f"✗ Error processing graph {idx+1}: {e}")
            continue

    # Overall metrics
    if all_act:
        overall_mae, overall_nmae = compute_mae_metrics(np.array(all_pred), np.array(all_act))
        print(f"\n{'='*80}")
        print(f"OVERALL RESULTS:")
        print(f"  Total predictions: {len(all_pred)}")
        print(f"  Overall MAE:  {overall_mae:.6f}")
        print(f"  Overall NMAE: {overall_nmae:.6f} ({overall_nmae*100:.2f}%)")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"RESULTS:")
        print(f"  Total predictions: {len(all_pred)}")
        print(f"  No ground truth available - metrics skipped")
        print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Inline single-graph demo
# ---------------------------------------------------------------------------

def run_inline_demo():
    """Run demo with hardcoded example graph."""
    print(f"\n{'='*80}")
    print("INLINE DEMO - Single Graph Prediction")
    print(f"{'='*80}")
    
    # Example 4-node network
    example_adj_matrix = [
        [0, 1, 0, 0],  # Node 0 connects to Node 1
        [1, 0, 1, 0],  # Node 1 connects to Nodes 0, 2
        [0, 1, 0, 1],  # Node 2 connects to Nodes 1, 3
        [0, 0, 1, 0],  # Node 3 connects to Node 2
    ]
    example_tp = [0.10, 0.07, 0.05, 0.08]
    
    print(f"Graph structure:")
    print(f"  Adjacency matrix: {example_adj_matrix}")
    print(f"  Transmission probabilities: {example_tp}")
    
    # Predict
    predictions = predict_saturation_throughput(example_adj_matrix, example_tp)
    
    print(f"\nPredicted saturation throughput:")
    for i, pred in enumerate(predictions):
        print(f"  Node {i}: {pred:.6f}")
    
    print(f"\nNetwork summary:")
    print(f"  Average transmission prob: {np.mean(example_tp):.4f}")
    print(f"  Average predicted throughput: {np.mean(predictions):.6f}")
    print(f"  Min predicted throughput: {np.min(predictions):.6f}")
    print(f"  Max predicted throughput: {np.max(predictions):.6f}")
    
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Enhanced GCN Saturation Throughput Predictor")
    print(f"Model architecture: {INPUT_DIM} → {HIDDEN_DIMS} → {OUTPUT_DIM}")
    
    if CSV_PATH and Path(CSV_PATH).exists():
        run_csv_evaluation(CSV_PATH)
    elif CSV_PATH:
        print(f"\n⚠ CSV file not found: {CSV_PATH}")
        print("Running inline demo instead...")
        run_inline_demo()
    else:
        run_inline_demo()
    
    # ---- Additional debug function ----
    def debug_prediction_vs_training():
        """Debug function to check prediction vs training data format."""
        print(f"\n{'='*60}")
        print("DEBUG: Checking prediction vs training format")
        print(f"{'='*60}")
        
        # Test with first graph from CSV if available
        if CSV_PATH and Path(CSV_PATH).exists():
            df = pd.read_csv(CSV_PATH)
            if len(df) > 0:
                first_row = df.iloc[0]
                test_adj = ast.literal_eval(first_row["adj_matrix"])
                test_tp = ast.literal_eval(first_row["transmission_prob"])
                
                print(f"Test graph: {len(test_tp)} nodes, {sum(sum(row) for row in test_adj)} edges")
                print(f"Transmission probs: {test_tp[:3]}... (showing first 3)")
                
                # Predict
                pred = predict_saturation_throughput(test_adj, test_tp)
                print(f"Predictions: {pred[:3]}... (showing first 3)")
                print(f"Prediction range: [{pred.min():.6f}, {pred.max():.6f}]")
                
                # Check against actual if available
                if "saturation_throughput" in first_row and pd.notna(first_row["saturation_throughput"]):
                    actual = ast.literal_eval(first_row["saturation_throughput"])
                    actual = np.array(actual)
                    print(f"Actual range: [{actual.min():.6f}, {actual.max():.6f}]")
                    
                    mae = np.mean(np.abs(pred - actual))
                    print(f"Sample MAE: {mae:.6f}")
                    
                    if mae > 0.01:  # High error
                        print("⚠️ High error detected! Possible issues:")
                        print("  - Model architecture mismatch")
                        print("  - Wrong model file loaded") 
                        print("  - Data format difference")
                else:
                    print("No ground truth available for comparison")
        
        print(f"{'='*60}")
    
    # Run debug
    debug_prediction_vs_training()


# ---------------------------------------------------------------------------
# Additional utility functions (if needed)
# ---------------------------------------------------------------------------

def predict_single_graph(adj_matrix: list[list[int]], 
                         transmission_prob: list[float],
                         verbose: bool = True) -> np.ndarray:
    """Convenience function for single graph prediction."""
    if verbose:
        print(f"Predicting throughput for {len(transmission_prob)}-node graph...")
    
    predictions = predict_saturation_throughput(adj_matrix, transmission_prob)
    
    if verbose:
        for i, (tp, pred) in enumerate(zip(transmission_prob, predictions)):
            print(f"Node {i}: p_i={tp:.3f} → throughput={pred:.6f}")
    
    return predictions


def batch_predict_from_lists(adj_matrices: list,
                            transmission_probs: list) -> list[np.ndarray]:
    """Predict throughput for multiple graphs."""
    results = []
    for i, (adj, tp) in enumerate(zip(adj_matrices, transmission_probs)):
        print(f"Processing graph {i+1}/{len(adj_matrices)}...")
        pred = predict_saturation_throughput(adj, tp)
        results.append(pred)
    return results