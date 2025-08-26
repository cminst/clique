
# 5_compare_gcn_pools_acc_norm_multiseed_v2.py
# Compare Plain GCN, GCN+L-RMC (seeded pooling), DiffPool, and "gPool" (TopK-style)
# on Cora node classification — with accuracy-based early stopping, row-normalized
# features, multi-seed runs, and a stronger skip path for L-RMC to mitigate
# mixed clusters. If you have an L-RMC seeds JSON, pass it via --seeds_json.

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, DenseGCNConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj

# -------------------------- Repro helpers --------------------------

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------- Metrics --------------------------

def macro_f1_from_logits(logits: Tensor, y: Tensor, mask: Tensor) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        y = y[mask]
        p = pred[mask]
        C = int(y.max().item() + 1)
        # confusion matrix
        cm = torch.zeros((C, C), dtype=torch.long, device=logits.device)
        for t, q in zip(y, p):
            cm[t, q] += 1
        tp = cm.diag().to(torch.float)
        fp = cm.sum(dim=0).to(torch.float) - tp
        fn = cm.sum(dim=1).to(torch.float) - tp
        prec = tp / (tp + fp).clamp(min=1.0)
        rec = tp / (tp + fn).clamp(min=1.0)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
        present = cm.sum(dim=1) > 0
        return f1[present].mean().item() if present.any() else 0.0

def accuracy_from_logits(logits: Tensor, y: Tensor, mask: Tensor) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        correct = (pred[mask] == y[mask]).sum().item()
        total = int(mask.sum().item())
        return correct / max(total, 1)

# -------------------------- Data --------------------------

def load_planetoid_cora(normalize: bool = True) -> Data:
    if normalize:
        dataset = Planetoid(root="/tmp/Cora", name="Cora", transform=T.NormalizeFeatures())
    else:
        dataset = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset[0]
    data.num_classes = dataset.num_classes
    return data

def make_planetoid_style_split(y: Tensor, num_classes: int, train_per_class=20, val_size=500, test_size=1000, seed: int = 0):
    # Random Planetoid-style split (reshuffled each call with given seed)
    set_seed(seed)
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
    k_val = min(val_size, remaining.numel())
    val_mask[remaining[:k_val]] = True
    remaining = remaining[k_val:]
    k_test = min(test_size, remaining.numel())
    test_mask[remaining[:k_test]] = True
    return train_mask, val_mask, test_mask

# -------------------------- Partitions & pooling --------------------------

def pool_by_partition(x: Tensor, edge_index: Tensor, cluster_id: Tensor, K: int) -> Tuple[Tensor, Tensor]:
    """
    Average node features per cluster and create the pooled edge_index among clusters.
    """
    N, F = x.size(0), x.size(1)
    device = x.device
    # aggregate features
    sums = torch.zeros((K, F), device=device, dtype=x.dtype)
    sums.index_add_(0, cluster_id, x)
    counts = torch.bincount(cluster_id, minlength=K).clamp_min(1).to(device).unsqueeze(1).to(x.dtype)
    x_pooled = sums / counts

    # pooled edges
    cu = cluster_id[edge_index[0]]
    cv = cluster_id[edge_index[1]]
    pairs = torch.stack([cu, cv], dim=1).tolist()
    uniq = set()
    pooled_edges = []
    for a, b in pairs:
        if a == b:
            continue
        key = (int(a), int(b))
        if key not in uniq:
            uniq.add(key)
            pooled_edges.append([key[0], key[1]])
    if pooled_edges:
        ei_pooled = torch.tensor(pooled_edges, dtype=torch.long, device=device).t().contiguous()
    else:
        ei_pooled = torch.empty((2, 0), dtype=torch.long, device=device)
    return x_pooled, ei_pooled

def cluster_majority_upper_bound(cluster_id: Tensor, y: Tensor) -> float:
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

def print_cluster_stats(cluster_id: Tensor, y: Tensor):
    ub = cluster_majority_upper_bound(cluster_id, y)
    K = int(cluster_id.max().item() + 1)
    N = y.size(0)
    print(f"Cluster count K = {K}, ratio K/N = {K/N:.3f}, majority-vote UB = {ub:.3f}")

def load_lrmc_partition_from_json(path: str, num_nodes: int) -> Tuple[Tensor, int]:
    """
    Expected format:
    {
      "clusters": [
        {"cluster_id": 0, "seed_nodes": [node_idx, ...]},
        {"cluster_id": 1, "seed_nodes": [...]},
        ...
      ]
    }
    Returns (cluster_id[N], K).
    Raises if some nodes are not covered.
    """
    obj = json.loads(Path(path).read_text())
    cid_of_node: Dict[int, int] = {}
    best = {}
    bestSc = -float('inf')
    for c in obj["clusters"]:
        cid = int(c["cluster_id"])
        if (c["score"]>bestSc):
            bestSc = c["score"]
            best = c["seed_nodes"]
        for u in c["seed_nodes"]:
            cid_of_node[int(u)] = cid
    cluster_id = torch.full((num_nodes,), -1, dtype=torch.long)
    for u, cid in cid_of_node.items():
        if 0 <= u < num_nodes:
            cluster_id[u] = cid
    if (cluster_id < 0).any():
        missing = int((cluster_id < 0).sum().item())
        raise RuntimeError(f"{missing} nodes not covered by seeds in {path}.")
    K = int(cluster_id.max().item() + 1)
    print(best)
    return cluster_id, K

def fallback_partition_topk_degree(edge_index: Tensor, N: int, K: int, device) -> Tensor:
    """
    Fallback seed selection: pick K highest-degree nodes as seeds, then assign each
    non-seed to the neighboring seed with highest degree (or to the global max-degree
    seed if isolated). This mimics the structure used in our TopK-based 'gPool' model.
    """
    deg = torch.bincount(edge_index[0], minlength=N).to(device)
    kept = torch.topk(deg, min(K, N), sorted=False).indices
    keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
    keep_mask[kept] = True

    neigh = [[] for _ in range(N)]
    u_list, v_list = edge_index[0].tolist(), edge_index[1].tolist()
    for a, b in zip(u_list, v_list):
        neigh[a].append(b)
        neigh[b].append(a)

    cluster_id = torch.full((N,), -1, dtype=torch.long, device=device)
    cluster_id[kept] = torch.arange(kept.numel(), device=device, dtype=torch.long)
    # tie-breaker
    best_global_kept = kept[torch.argmax(deg[kept])].item() if kept.numel() > 0 else 0
    for u in range(N):
        if keep_mask[u]:
            continue
        cand = [w for w in neigh[u] if keep_mask[w]]
        if cand:
            # pick the neighbor seed with highest degree
            cluster_id[u] = cluster_id[max(cand, key=lambda z: int(deg[z].item()))]
        else:
            cluster_id[u] = cluster_id[best_global_kept]
    return cluster_id

# -------------------------- Models --------------------------

class PlainGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x, 0.0

class LrmcSeededPoolGCN(nn.Module):
    """
    Single-shot seeded pooling by an L-RMC partition.
    Pooled logits are broadcast back and combined with a strengthened skip head.
    We add a LEARNABLE scalar alpha >= 0 for the skip head to allow the model
    to override bad cluster majorities during training.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, cluster_id: Tensor, K: int, dropout=0.5, alpha_init=1.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        self.lin_skip = nn.Linear(hidden_dim, out_dim, bias=True)
        # learnable, non-negative scaling for skip head
        self.alpha = nn.Parameter(torch.tensor([alpha_init], dtype=torch.float))
        self.alpha_relu = nn.ReLU()
        self.dropout = dropout
        self.register_buffer("cluster_id", cluster_id)
        self.K = K

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        # pool and conv on pooled graph
        x_p, ei_p = pool_by_partition(x1, edge_index, self.cluster_id, self.K)
        x_p = self.conv2(x_p, ei_p)           # [K, C]
        up = x_p[self.cluster_id]             # [N, C]
        skip = self.lin_skip(x1)              # [N, C]
        logits = up + self.alpha_relu(self.alpha) * skip
        return logits, 0.0

class TopKPoolBroadcastGCN(nn.Module):
    """
    A simple "gPool"-style baseline:
    1) GCNConv -> scores -> keep top-K nodes
    2) Assign every non-kept node to a neighboring kept node (highest-degree tie-breaker)
    3) Pool by partition, run a GCN on K-node graph, then broadcast + skip
    """
    def __init__(self, in_dim, hidden_dim, out_dim, K_target: int, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        self.lin_skip = nn.Linear(hidden_dim, out_dim, bias=True)
        self.dropout = dropout
        self.K_target = K_target
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, edge_index):
        device = x.device
        N = x.size(0)
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        raw = self.score(x1).squeeze(-1)                # [N]
        gate = torch.tanh(raw).unsqueeze(-1)            # mild gating
        x1_gated = x1 * gate
        # degrees for fallback assignment
        deg = torch.bincount(edge_index[0], minlength=N).to(device)
        K = min(self.K_target, N)
        kept = torch.topk(raw, K, sorted=True).indices
        keep_mask = torch.zeros(N, dtype=torch.bool, device=device); keep_mask[kept] = True
        # adjacency list
        u_list, v_list = edge_index[0].tolist(), edge_index[1].tolist()
        neigh = [[] for _ in range(N)]
        for a, b in zip(u_list, v_list):
            neigh[a].append(b); neigh[b].append(a)
        # build partition
        cluster_id = torch.full((N,), -1, dtype=torch.long, device=device)
        cluster_id[kept] = torch.arange(kept.numel(), device=device, dtype=torch.long)
        best_global_kept = kept[torch.argmax(deg[kept])].item() if kept.numel() > 0 else 0
        for u in range(N):
            if keep_mask[u]: continue
            cand = [w for w in neigh[u] if keep_mask[w]]
            cluster_id[u] = cluster_id[max(cand, key=lambda z: int(deg[z].item()))] if cand else cluster_id[best_global_kept]
        Kc = int(cluster_id.max().item() + 1)
        # pool
        x_p, ei_p = pool_by_partition(x1_gated, edge_index, cluster_id, Kc)
        x_p = self.conv2(x_p, ei_p)
        up = x_p[cluster_id]
        skip = self.lin_skip(x1)
        logits = up + skip
        return logits, 0.0

class DiffPoolOneShot(nn.Module):
    """
    Minimal one-layer DiffPool for a single-graph dataset like Cora.
    We embed and assign once, apply dense_diff_pool, then run a small GCN on pooled graph.
    Broadcast back the pooled logits with a skip connection from the pre-pool features.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, K: int, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.gnn_embed1 = DenseGCNConv(in_dim, hidden_dim)
        self.gnn_embed2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.gnn_assign1 = DenseGCNConv(in_dim, hidden_dim)
        self.gnn_assign2 = DenseGCNConv(hidden_dim, K)
        self.gnn_post1 = DenseGCNConv(hidden_dim, hidden_dim)
        self.gnn_post2 = DenseGCNConv(hidden_dim, out_dim)
        self.lin_skip = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x, edge_index):
        device = x.device
        N = x.size(0)
        adj = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0).to(device)  # [N, N]
        x = x.to(device)
        mask = torch.ones((1, N), device=device)  # single graph mask
        # Embed and assign
        z = F.relu(self.gnn_embed1(x.unsqueeze(0), adj.unsqueeze(0), mask))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.relu(self.gnn_embed2(z, adj.unsqueeze(0), mask))
        s = F.relu(self.gnn_assign1(x.unsqueeze(0), adj.unsqueeze(0), mask))
        s = F.dropout(s, p=self.dropout, training=self.training)
        s = self.gnn_assign2(s, adj.unsqueeze(0), mask).softmax(dim=-1)  # [1, N, K]
        # DiffPool
        x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(z, adj.unsqueeze(0), s, mask)
        # Post-pool GNN
        h = F.relu(self.gnn_post1(x_pool, adj_pool))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gnn_post2(h, adj_pool)                       # [1, K, C]
        # Broadcast with skip from pre-pool feature (z.squeeze(0))
        skip = self.lin_skip(z.squeeze(0))                    # [N, C]
        logits_nodes = torch.matmul(s.squeeze(0), h.squeeze(0)) + skip
        aux_loss = link_loss + ent_loss
        return logits_nodes, aux_loss

# -------------------------- Training --------------------------

def train_one(model: nn.Module, data: Data, train_mask: Tensor, val_mask: Tensor, test_mask: Tensor, device,
              epochs=400, patience=50, lr=1e-2, weight_decay=5e-4, val_metric="accuracy", aux_weight=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = -1.0
    best_state = None
    best_test = None

    def get_scores(logits):
        if val_metric == "macro_f1":
            return macro_f1_from_logits(logits, data.y, val_mask), accuracy_from_logits(logits, data.y, test_mask), macro_f1_from_logits(logits, data.y, test_mask)
        else:
            # accuracy validation
            return accuracy_from_logits(logits, data.y, val_mask), accuracy_from_logits(logits, data.y, test_mask), macro_f1_from_logits(logits, data.y, test_mask)

    bad = 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, aux_loss = model(data.x.to(device), data.edge_index.to(device))
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask].to(device)) + aux_weight * aux_loss
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(data.x.to(device), data.edge_index.to(device))
            val_score, test_acc, test_f1 = get_scores(logits)
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test = (float(test_acc), float(test_f1))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # restore
    if best_state is not None:
        model.load_state_dict(best_state)
    # final test (best saved)
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x.to(device), data.edge_index.to(device))
        test_acc = accuracy_from_logits(logits, data.y, test_mask)
        test_f1 = macro_f1_from_logits(logits, data.y, test_mask)

    # prefer stored best_test if present
    if best_test is not None:
        test_acc, test_f1 = best_test
    return {"test_acc": test_acc, "test_f1": test_f1}

# -------------------------- Orchestration --------------------------

def run_once(args, seed: int):
    set_seed(seed)

    # Load data (fixed Planetoid split by default)
    data = load_planetoid_cora(normalize=True)
    if not args.use_planetoid_split:
        tr, va, te = make_planetoid_style_split(data.y, data.num_classes, seed=seed)
        data.train_mask, data.val_mask, data.test_mask = tr, va, te

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    # Determine K and cluster assignment for L-RMC
    N = data.num_nodes
    K = args.K if args.K is not None else max(2, int(args.k_ratio * N))
    if args.seeds_json is not None and Path(args.seeds_json).exists():
        cluster_id, K_lrmc = load_lrmc_partition_from_json(args.seeds_json, N)
        if args.K is not None and K_lrmc != K:
            print(f"[warn] K from seeds ({K_lrmc}) differs from requested K ({K}). Using seeds K.")
        K = K_lrmc
    else:
        print("[info] No seeds_json provided or file not found — using TopK-degree fallback partition.")
        cluster_id = fallback_partition_topk_degree(data.edge_index, N, K, device=data.x.device)

    print_cluster_stats(cluster_id, data.y)

    # Instantiate models
    base_model = PlainGCN(data.num_features, args.hidden, data.num_classes, dropout=args.dropout)
    lrmc_model = LrmcSeededPoolGCN(data.num_features, args.hidden, data.num_classes, cluster_id=cluster_id.to(data.x.device), K=K, dropout=args.dropout, alpha_init=args.alpha_init)
    diff_model = DiffPoolOneShot(data.num_features, args.hidden, data.num_classes, K=K, dropout=args.dropout)
    gpool_model = TopKPoolBroadcastGCN(data.num_features, args.hidden, data.num_classes, K_target=K, dropout=args.dropout)

    # Train each
    cfg = dict(epochs=args.epochs, patience=args.patience, lr=args.lr, weight_decay=args.weight_decay, val_metric=args.val_metric, aux_weight=args.diffpool_aux_weight)
    res_base = train_one(base_model, data, train_mask, val_mask, test_mask, device, **cfg)
    res_lrmc = train_one(lrmc_model, data, train_mask, val_mask, test_mask, device, **cfg)
    res_diff = train_one(diff_model, data, train_mask, val_mask, test_mask, device, **cfg)
    res_gpool = train_one(gpool_model, data, train_mask, val_mask, test_mask, device, **cfg)

    return res_base, res_lrmc, res_diff, res_gpool

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds_json", type=str, default=None, help="Path to L-RMC seeds JSON (clusters with member node IDs).")
    p.add_argument("--K", type=int, default=None, help="If provided, number of clusters (overrides k_ratio when seeds_json is absent).")
    p.add_argument("--k_ratio", type=float, default=0.5, help="Target K/N when --K is not provided. Higher -> smaller clusters.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--val_metric", type=str, default="accuracy", choices=["accuracy", "macro_f1"], help="Early stopping metric (default: accuracy)")
    p.add_argument("--diffpool_aux_weight", type=float, default=1e-3)
    p.add_argument("--alpha_init", type=float, default=1.5, help="Initial value for L-RMC skip scaling (learnable, non-negative)")
    p.add_argument("--use_planetoid_split", action="store_true", help="Use the fixed Planetoid train/val/test masks.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument("--seeds", type=str, default="42-59", help="Seed list or range, e.g., '1,2,3' or '42-59'")

    args = p.parse_args()

    # Parse seeds
    seed_list = []
    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seed_list = list(range(int(a), int(b) + 1))
    else:
        seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Run multi-seed
    all_res = {"PlainGCN": [], "L-RMC": [], "DiffPool": [], "gPool": []}
    for s in seed_list:
        print(f"\n==== Running seed {s} ====")
        res_base, res_lrmc, res_diff, res_gpool = run_once(args, seed=s)
        print(f"Seed {s} | PlainGCN acc {res_base['test_acc']:.3f} | L-RMC {res_lrmc['test_acc']:.3f} | DiffPool {res_diff['test_acc']:.3f} | gPool {res_gpool['test_acc']:.3f}")
        for (name, res) in [("PlainGCN", res_base), ("L-RMC", res_lrmc), ("DiffPool", res_diff), ("gPool", res_gpool)]:
            all_res[name].append((res["test_acc"], res["test_f1"]))

    def mean_std(vals):
        import statistics as st
        m = st.mean(vals) if len(vals) > 0 else float("nan")
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        return m, sd

    print(f"\n===== Multi-seed Summary (mean ± std over {len(seed_list)} seeds) =====")
    for name in ["PlainGCN", "L-RMC", "DiffPool", "gPool"]:
        accs = [x[0] for x in all_res[name]]
        f1s  = [x[1] for x in all_res[name]]
        am, asd = mean_std(accs)
        fm, fsd = mean_std(f1s)
        print(f"{name:9s} | acc {am:.3f} ± {asd:.3f} | macro-F1 {fm:.3f} ± {fsd:.3f}")

if __name__ == "__main__":
    main()
