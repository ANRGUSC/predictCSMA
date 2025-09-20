
# ------------------------- 0) IMPORTS --------------------------------------
import torch, torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv          # only needed for edge attr
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, DataLoader
import numpy as np, pandas as pd, ast, random, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from GNN_Architecture import GINNet 

# ------------------------- 1) GLOBAL SETTINGS ------------------------------
T          = 5                
CSV_PATH   = 'Data/5_aggregated_data_million.csv'
BATCH_SIZE = 32
EPOCHS     = 200
SEED       = 2                 # optional: set None for non-deterministic run
print(T)
# ------------------------- 2) (Optional) REPRODUCIBILITY -------------------
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ------------------------- 3) DATA -> torch_geometric.Data -----------------
def create_data(p_vec, adj_mat, sat_tput,theo):

    # build directed edge list
    edges = [(i, j) for i, row in enumerate(adj_mat)
                     for j, val in enumerate(row) if val == 1]
    if not edges:
        return None
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # self-loops (GIN itself ignores edge_attr, but loops help consistency)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(p_vec))

    x = torch.tensor(np.column_stack([p_vec]), dtype=torch.float)
    y = torch.tensor(sat_tput, dtype=torch.float).view(-1, 1)

    return Data(x=x, edge_index=edge_index, y=y)

def load_dataset(path: str):
    df = pd.read_csv(path)
    data_lst = []; adjs = []; probs = []
    for _, r in df.iterrows():
        G   = ast.literal_eval(r['adj_matrix'])
        p   = ast.literal_eval(r['transmission_prob'])
        sat = ast.literal_eval(r['saturation_throughput'])
        theo = ast.literal_eval(r['theoretical_throughput'])
        d = create_data(p, G, sat,theo)
        if d is not None:
            data_lst.append(d); adjs.append(G); probs.append(p)
    return data_lst, adjs, probs

data_list, *_ = load_dataset(CSV_PATH)
print('Example Data object:', data_list[0])

# ------------------------- 4) SPLIT & LOADERS ------------------------------
tr, te = train_test_split(data_list, test_size=0.2,
                          random_state=SEED, shuffle=True)
tr, va = train_test_split(tr, test_size=0.1 / 0.8,
                          random_state=SEED, shuffle=True)

train_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(va, batch_size=BATCH_SIZE)
test_loader  = DataLoader(te, batch_size=BATCH_SIZE)
print(f'Train/Val/Test graphs: {len(tr)}/{len(va)}/{len(te)}')

# ------------------------- 5) MODEL / OPTIMISER ----------------------------
model       = GINNet(input_dim=1, hidden_dims=[64]*4, output_dim=1)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params:,}')

opt        = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',
                                                        factor=0.5, patience=5)
criterion   = nn.MSELoss()

# --------------------------------------------------------------------------- #
# 5) EVALUATION                              
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
for epoch in range(250):
    model.train(); tot = 0
    for batch in train_loader:
        opt.zero_grad()
        loss = criterion(model(batch), batch.y)
        loss.backward(); opt.step()
        tot += loss.item() * batch.num_graphs

    tr_loss = tot / len(train_loader.dataset)
    val_loss, val_mae, val_nmae, *_ = evaluate(model, val_loader)
    scheduler.step(val_loss)

    if val_loss < best:
        best = val_loss
        torch.save(model.state_dict(), f'{T}_bestGIN.pt')

    if (epoch + 1) % 10 == 0:
        print(f'E{epoch+1:03d} | Train {tr_loss:.4f} | Val {val_loss:.4f} | '
              f'MAE {val_mae:.4f} | NMAE {val_nmae:.4f}')

# --------------------------------------------------------------------------- #
# 7) FINAL EVALUATION                                                         #
# --------------------------------------------------------------------------- #
model.load_state_dict(torch.load(f'{T}_bestGIN.pt'))
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