
#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
#  GCN — node features = [ p_i ]
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd, ast, random, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from GNN_Architecture import GCN, DGCNConv 
# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
T=7
CSV_PATH   = 'Data/5_data_million.csv'
BATCH_SIZE = 32
EPOCHS     = 200
SEED       = 2
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

import networkx as nx
# --------------------------------------------------------------------------- #
# 1) CSV  →  torch_geometric.Data                                             #
# --------------------------------------------------------------------------- #
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

def load_dataset(path):
    df = pd.read_csv(path)
    data = []
    for _, r in df.iterrows():
        G   = ast.literal_eval(r['adj_matrix'])
        p   = ast.literal_eval(r['transmission_prob'])
        sat = ast.literal_eval(r['saturation_throughput'])
        # theo = ast.literal_eval(r['theoretical_throughput'])
        # d = make_data(G, p, sat, theo)
        d = make_data(G, p, sat)
        if d is not None:
            data.append(d)
    return data

dataset = load_dataset(CSV_PATH)
train, test = train_test_split(dataset, test_size=0.2, random_state=SEED)
train, val  = train_test_split(train, test_size=0.1/0.8, random_state=SEED)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test,  batch_size=BATCH_SIZE)

# --------------------------------------------------------------------------- #
# 2) MODEL / OPTIMISER / LOSS                                                 #
# --------------------------------------------------------------------------- #
input_dim   = 1                    # ← ONLY p_i
hidden_dims = [64]*7
output_dim  = 1
model       = DGCNConv(input_dim, hidden_dims, output_dim)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params:,}')

opt        = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)


scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',
                                                        factor=0.5, patience=5)
criterion   = nn.MSELoss()

# --------------------------------------------------------------------------- #
# 5) EVALUATION (no theoretical column anymore)                               #
# --------------------------------------------------------------------------- #
def evaluate(model, loader):
    model.eval()
    tot_loss, preds, acts = 0, [], []

    with torch.no_grad():
        for d in loader:
            out   = model(d)
            loss  = criterion(out, d.y)
            tot_loss += loss.item()

            preds.append((out).cpu().numpy())
            acts.append((d.y).cpu().numpy())

    pred, act = map(np.vstack, (preds, acts))
    mae  = np.mean(np.abs(pred - act))
    nmae = mae / np.mean(act) if np.mean(act) > 0 else 0
    return tot_loss / len(loader), mae, nmae, pred, act


# --------------------------------------------------------------------------- #
# 6) TRAINING LOOP                                                            #
# --------------------------------------------------------------------------- #
best = float('inf')
for epoch in range(EPOCHS):
    model.train(); tot = 0
    for batch in train_loader:
        opt.zero_grad()
        loss = criterion(model(batch), batch.y)
        loss.backward(); 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        

        tot += loss.item() * batch.num_graphs

    tr_loss = tot / len(train_loader.dataset)
    val_loss, val_mae, val_nmae, *_ = evaluate(model, val_loader)
    scheduler.step(val_loss)

    if val_loss < best:
        best = val_loss
        torch.save(model.state_dict(), f'{T}_bestenhanced.pt')

    if (epoch + 1) % 10 == 0:
        print(f'E{epoch+1:03d} | Train {tr_loss:.4f} | Val {val_loss:.4f} | '
              f'MAE {val_mae:.4f} | NMAE {val_nmae:.4f}')

# --------------------------------------------------------------------------- #
# 7) FINAL EVALUATION                                                         #
# --------------------------------------------------------------------------- #
model.load_state_dict(torch.load(f'{T}_bestenhanced.pt'))
tr_metrics = evaluate(model, train_loader)
te_metrics = evaluate(model, test_loader)

print(f'\nTrain MAE {tr_metrics[1]:.4f} | NMAE {tr_metrics[2]:.4f}')
print(f'Test  MAE {te_metrics[1]:.4f} | NMAE {te_metrics[2]:.4f}')

# --------------------------------------------------------------------------- #
# 8) VISUALISATION                                                            #
# --------------------------------------------------------------------------- #
plt.figure(figsize=(10, 6))
plt.plot(te_metrics[4].flatten(), label='Actual')
plt.plot(te_metrics[3].flatten(), '--', label='Predicted')
plt.xlabel('Sample'); plt.ylabel('Throughput'); plt.legend()
plt.title('Actual vs Predicted Saturation Throughput (no Θ_theo feature)')
plt.show()

print("\nRandom test samples:")
pairs = list(zip(te_metrics[3].flatten(), te_metrics[4].flatten()))
for i, (p, a) in enumerate(random.sample(pairs, 10), 1):
    err = abs(p - a); pct = err / a * 100 if a > 0 else 0
    print(f"{i:2d} | {p:.6f} | {a:.6f} | {err:.6f} | {pct:5.1f}%")