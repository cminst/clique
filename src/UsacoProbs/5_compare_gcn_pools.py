# compare_pools.py
# Compare GCN+L-RMC, GCN+DiffPool, and GCN+gPool on Cora node classification.

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import add_self_loops

# DiffPool bits (dense)
from torch_geometric.nn import DenseGCNConv, dense_diff_pool

# ---------------- Config ----------------
SEEDS_JSON = "clique_tests/seeds/seeds_diam_1e-8.json"
CORA_CONTENT = "cora/cora.content"
CORA_CITES = "cora/cora.cites"

SEED = 42
HIDDEN = 64
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 400
PATIENCE = 50
VAL_METRIC = "macro_f1"  # "macro_f1" or "accuracy"
DIFFPOOL_AUX_WEIGHT = 1e-3  # weight for link + entropy losses

# --------------- Utils ------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_undirected(edge_index, num_nodes):
    # edge_index [2, E], return unique undirected edges without self loops
    ei = edge_index
    rev = torch.stack([ei[1], ei[0]], dim=0)
    ei2 = torch.cat([ei, rev], dim=1)
    # dedup via set of tuples
    cols = ei2.t().tolist()
    uniq = set()
    out = []
    for u, v in cols:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in uniq:
            uniq.add(key)
            out.append([key[0], key[1]])
    if not out:
        return torch.empty((2, 0), dtype=torch.long)
    out = torch.tensor(out, dtype=torch.long).t().contiguous()
    return out

def macro_f1_from_logits(logits, y, mask):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        y = y[mask]
        p = pred[mask]
        C = int(y.max().item() + 1)
        cm = torch.zeros((C, C), dtype=torch.long, device=logits.device)
        for t, q in zip(y, p):
            cm[t, q] += 1
        eps = 1e-12
        tp = cm.diag().to(torch.float)
        fp = cm.sum(dim=0).to(torch.float) - tp
        fn = cm.sum(dim=1).to(torch.float) - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        present = cm.sum(dim=1) > 0
        return f1[present].mean().item() if present.any() else 0.0

def accuracy_from_logits(logits, y, mask):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        correct = (pred[mask] == y[mask]).sum().item()
        total = int(mask.sum().item())
        return correct / max(total, 1)

# ------------- Load Cora ----------------
def load_cora_from_content_and_cites(content_path: str, cites_path: str):
    lines = Path(content_path).read_text().strip().splitlines()
    n = len(lines)

    paper_ids, features, labels_raw = [], [], []
    for line in lines:
        toks = line.strip().split()
        paper_ids.append(toks[0])
        labels_raw.append(toks[-1])
        feat = [int(x) for x in toks[1:-1]]
        features.append(feat)

    classes = sorted(set(labels_raw))
    cls2idx = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([cls2idx[c] for c in labels_raw], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)

    id2idx = {pid: i for i, pid in enumerate(paper_ids)}
    edges = []
    for line in Path(cites_path).read_text().strip().splitlines():
        a, b = line.strip().split()
        if a in id2idx and b in id2idx:
            edges.append((id2idx[a], id2idx[b]))
    if not edges:
        raise RuntimeError("No edges from cites file.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, n)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = n
    data.num_classes = len(classes)
    return data

def make_planetoid_style_split(y, num_classes, train_per_class=20, val_size=500, test_size=1000):
    N = y.size(0)
    all_idx = torch.arange(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    for c in range(num_classes):
        idx_c = all_idx[(y == c)]
        if idx_c.numel() == 0:
            continue
        sel = idx_c[torch.randperm(idx_c.numel())[: min(train_per_class, idx_c.numel())]]
        train_mask[sel] = True

    remaining = all_idx[~train_mask]
    remaining = remaining[torch.randperm(remaining.numel())]

    val_k = min(val_size, remaining.numel())
    val_mask[remaining[:val_k]] = True

    rem2 = remaining[val_k:]
    test_k = min(test_size, rem2.numel())
    test_mask[rem2[:test_k]] = True
    return train_mask, val_mask, test_mask

# ----------- L-RMC seeds pooling ----------
def load_lrmc_partition(path: str, num_nodes: int):
    obj = json.loads(Path(path).read_text())
    clusters = obj["clusters"]
    cid_of_node = {}
    for c in clusters:
        cid = int(c["cluster_id"])
        for u in c["members"]:
            cid_of_node[int(u)] = cid
    cluster_id = torch.full((num_nodes,), -1, dtype=torch.long)
    for u, cid in cid_of_node.items():
        if 0 <= u < num_nodes:
            cluster_id[u] = cid
    if (cluster_id < 0).any():
        missing = int((cluster_id < 0).sum().item())
        raise RuntimeError(f"{missing} nodes not covered by seeds.")
    K = int(cluster_id.max().item() + 1)
    return cluster_id, K

def pool_by_partition(x, edge_index, cluster_id, K):
    N, F = x.size(0), x.size(1)
    sums = torch.zeros((K, F), device=x.device, dtype=x.dtype)
    sums.index_add_(0, cluster_id, x)
    counts = torch.bincount(cluster_id, minlength=K).clamp_min(1).to(x.device).unsqueeze(1).to(x.dtype)
    x_pooled = sums / counts

    cu = cluster_id[edge_index[0]]
    cv = cluster_id[edge_index[1]]
    pairs = torch.stack([cu, cv], dim=1)
    # dedup and drop self loops
    pairs = pairs.tolist()
    uniq = set()
    out = []
    for a, b in pairs:
        if a == b:
            continue
        key = (int(a), int(b))
        if key not in uniq:
            uniq.add(key)
            out.append([key[0], key[1]])
    edge_index_pooled = torch.tensor(out, device=x.device, dtype=torch.long).t().contiguous() if out else torch.empty((2, 0), dtype=torch.long, device=x.device)
    return x_pooled, edge_index_pooled

def cluster_majority_upper_bound(cluster_id, y):
    K = int(cluster_id.max().item() + 1)
    correct = 0
    for k in range(K):
        idx = (cluster_id == k)
        ys = y[idx]
        if ys.numel() == 0:
            continue
        _, counts = torch.unique(ys, return_counts=True)
        correct += int(counts.max().item())
    return correct / y.size(0)

def print_cluster_stats(cluster_id, y):
    ub = cluster_majority_upper_bound(cluster_id, y)
    K = int(cluster_id.max().item() + 1)
    N = y.size(0)
    print(f"Cluster count K = {K}, ratio K/N = {K/N:.3f}, "
          f"majority-vote upper bound = {ub:.3f}")

# ----------- Models -----------------------
class LrmcSeededPoolGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, cluster_id, K, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        self.lin_skip = nn.Linear(hidden_dim, out_dim, bias=True)
        self.dropout = dropout
        self.register_buffer("cluster_id", cluster_id)
        self.K = K
    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        # Pool once
        x_p, ei_p = pool_by_partition(x1, edge_index, self.cluster_id, self.K)
        x_p = self.conv2(x_p, ei_p)                  # [K, C]
        up = x_p[self.cluster_id]                    # [N, C]
        skip = self.lin_skip(x1)                     # [N, C]
        logits = up + skip
        return logits, 0.0

class TopKPoolBroadcastGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, K_target, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        self.lin_skip = nn.Linear(hidden_dim, out_dim, bias=True)
        self.dropout = dropout
        self.K_target = K_target
        self.score = nn.Linear(hidden_dim, 1, bias=False)
    @staticmethod
    def _degrees(edge_index, num_nodes):
        return torch.bincount(edge_index[0], minlength=num_nodes).to(torch.long)
    def forward(self, x, edge_index):
        N = x.size(0)
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        raw = self.score(x1).squeeze(-1)
        gate = torch.tanh(raw).unsqueeze(-1)
        x1_gated = x1 * gate
        K = min(self.K_target, N)
        kept = torch.topk(raw, K, sorted=True).indices
        keep_mask = torch.zeros(N, dtype=torch.bool, device=x.device); keep_mask[kept] = True
        deg = self._degrees(edge_index, N).to(x.device)
        u_list, v_list = edge_index[0].tolist(), edge_index[1].tolist()
        neigh = [[] for _ in range(N)]
        for a, b in zip(u_list, v_list):
            neigh[a].append(b); neigh[b].append(a)
        cluster_id = torch.full((N,), -1, dtype=torch.long, device=x.device)
        cluster_id[kept] = torch.arange(kept.numel(), device=x.device, dtype=torch.long)
        best_global_kept = kept[torch.argmax(deg[kept])].item() if kept.numel() > 0 else 0
        for u in range(N):
            if keep_mask[u]: continue
            cand = [w for w in neigh[u] if keep_mask[w]]
            cluster_id[u] = cluster_id[max(cand, key=lambda z: int(deg[z].item()))] if cand else cluster_id[best_global_kept]
        Kc = int(cluster_id.max().item() + 1)
        x_p, ei_p = pool_by_partition(x1_gated, edge_index, cluster_id, Kc)
        x_p = self.conv2(x_p, ei_p)
        up = x_p[cluster_id]
        skip = self.lin_skip(x1)
        logits = up + skip
        return logits, 0.0

class DiffPoolGCNNode(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, K_clusters, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.K = K_clusters
        self.gnn_embed1 = DenseGCNConv(in_dim, hidden_dim)
        self.gnn_embed2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.gnn_assign1 = DenseGCNConv(in_dim, hidden_dim)
        self.gnn_assign2 = DenseGCNConv(hidden_dim, K_clusters)
        self.gnn_post1  = DenseGCNConv(hidden_dim, hidden_dim)
        self.gnn_post2  = DenseGCNConv(hidden_dim, out_dim)
        self.lin_skip   = nn.Linear(hidden_dim, out_dim, bias=True)
    def forward(self, x, edge_index):
        N, device = x.size(0), x.device
        # Dense adjacency with self loops
        adj_dense = torch.zeros((N, N), device=device)
        adj_dense[edge_index[0], edge_index[1]] = 1.0
        adj_dense[torch.arange(N, device=device), torch.arange(N, device=device)] = 1.0
        x = x.unsqueeze(0)                    # [1, N, F]
        adj = adj_dense.unsqueeze(0)          # [1, N, N]
        mask = torch.ones((1, N), device=device)
        # Embed and assign
        z = F.relu(self.gnn_embed1(x, adj, mask))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.relu(self.gnn_embed2(z, adj, mask))
        s = F.relu(self.gnn_assign1(x, adj, mask))
        s = F.dropout(s, p=self.dropout, training=self.training)
        s = self.gnn_assign2(s, adj, mask).softmax(dim=-1)   # [1, N, K]
        # DiffPool
        x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        # Post-pool GNN
        h = F.relu(self.gnn_post1(x_pool, adj_pool))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gnn_post2(h, adj_pool)                       # [1, K, C]
        # Broadcast with skip from pre-pool features
        skip = self.lin_skip(z.squeeze(0))                    # [N, C]
        logits_nodes = torch.matmul(s.squeeze(0), h.squeeze(0)) + skip
        aux_loss = link_loss + ent_loss
        return logits_nodes, aux_loss

class PlainGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x, 0.0

# ------------- Training loop -------------
def train_one(model, data, train_mask, val_mask, test_mask, device, aux_weight=0.0):
    model = model.to(device)
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val = -math.inf
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logits, aux_loss = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        if aux_weight > 0.0:
            loss = loss + aux_weight * aux_loss
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(data.x, data.edge_index)
            if VAL_METRIC == "macro_f1":
                val_metric = macro_f1_from_logits(logits, data.y, val_mask)
            else:
                val_metric = accuracy_from_logits(logits, data.y, val_mask)

        if val_metric > best_val:
            best_val = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        train_acc = accuracy_from_logits(logits, data.y, train_mask)
        val_acc = accuracy_from_logits(logits, data.y, val_mask)
        test_acc = accuracy_from_logits(logits, data.y, test_mask)
        val_f1 = macro_f1_from_logits(logits, data.y, val_mask)
        test_f1 = macro_f1_from_logits(logits, data.y, test_mask)

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "val_f1": val_f1,
        "test_f1": test_f1,
    }

# ----------------- Main ------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Cora so indices match your seeds.json
    data = load_cora_from_content_and_cites(CORA_CONTENT, CORA_CITES)
    print(f"Loaded Cora: N={data.num_nodes}, E={data.edge_index.size(1)}, "
          f"F={data.num_features}, C={data.num_classes}")

    # Fixed Planetoid-style split for all runs
    train_mask, val_mask, test_mask = make_planetoid_style_split(
        data.y, data.num_classes, train_per_class=20, val_size=500, test_size=1000
    )

    # L-RMC partition and K
    cluster_id, K_lrmc = load_lrmc_partition(SEEDS_JSON, data.num_nodes)
    print(f"K_lrmc = {K_lrmc}")

    print_cluster_stats(cluster_id, data.y)

    # Optional diagnostic if you added the helper
    # cluster_purity(cluster_id, data.y)

    # 0) Plain 2-layer GCN baseline
    base_model = PlainGCN(
        in_dim=data.num_features,
        hidden_dim=HIDDEN,
        out_dim=data.num_classes,
        dropout=DROPOUT,
    )
    res_base = train_one(base_model, data, train_mask, val_mask, test_mask, device)
    print("\nPlain 2-layer GCN")
    print(f"Test acc {res_base['test_acc']:.3f} | Test macro-F1 {res_base['test_f1']:.3f}")

    # 1) GCN + L-RMC seeded pooling (with skip head)
    lrmc_model = LrmcSeededPoolGCN(
        in_dim=data.num_features,
        hidden_dim=HIDDEN,
        out_dim=data.num_classes,
        cluster_id=cluster_id,
        K=K_lrmc,
        dropout=DROPOUT,
    )
    res_lrmc = train_one(lrmc_model, data, train_mask, val_mask, test_mask, device)
    print("\nGCN + L-RMC seeded pooling")
    print(f"Test acc {res_lrmc['test_acc']:.3f} | Test macro-F1 {res_lrmc['test_f1']:.3f}")

    # 2) GCN + DiffPool (K matched, with skip head)
    diff_model = DiffPoolGCNNode(
        in_dim=data.num_features,
        hidden_dim=HIDDEN,
        out_dim=data.num_classes,
        K_clusters=K_lrmc,
        dropout=DROPOUT,
    )
    res_diff = train_one(
        diff_model, data, train_mask, val_mask, test_mask, device,
        aux_weight=DIFFPOOL_AUX_WEIGHT,  # try 1e-2 if training is wobbly
    )
    print("\nGCN + DiffPool (K matched)")
    print(f"Test acc {res_diff['test_acc']:.3f} | Test macro-F1 {res_diff['test_f1']:.3f}")

    # 3) GCN + gPool/TopK (K matched, with skip head)
    gpool_model = TopKPoolBroadcastGCN(
        in_dim=data.num_features,
        hidden_dim=HIDDEN,
        out_dim=data.num_classes,
        K_target=K_lrmc,
        dropout=DROPOUT,
    )
    res_gpool = train_one(gpool_model, data, train_mask, val_mask, test_mask, device)
    print("\nGCN + gPool/TopK (K matched)")
    print(f"Test acc {res_gpool['test_acc']:.3f} | Test macro-F1 {res_gpool['test_f1']:.3f}")

    # Summary
    print("\n===== Summary =====")
    for name, res in [
        ("PlainGCN", res_base),
        ("L-RMC", res_lrmc),
        ("DiffPool", res_diff),
        ("gPool", res_gpool),
    ]:
        print(f"{name:9s} | acc {res['test_acc']:.3f} | macro-F1 {res['test_f1']:.3f}")

if __name__ == "__main__":
    main()
