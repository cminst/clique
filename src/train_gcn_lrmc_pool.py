# sweep_lowlab_compress.py
# Low-label and compression sweeps for L-RMC, DiffPool, and gPool on Cora.

import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, DenseGCNConv
from torch_geometric.nn.dense import dense_diff_pool

# ============ Paths ============
SEEDS_JSON = "../seeds_diam_1e-6.json"   # your L-RMC export
CORA_CONTENT = "../cora/cora.content"
CORA_CITES = "../cora/cora.cites"

# ============ Sweep settings ============
LABEL_BUDGETS = [20, 10, 5, 3]        # train_per_class
K_RATIOS = [0.10, 0.20, 0.40, 0.80]   # K / N target
SEEDS = [0, 1, 2, 3, 4]               # random seeds per cell

# ============ Train hyperparams ============
HIDDEN = 64
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 300
PATIENCE = 50

# DiffPool extras
DIFFPOOL_AUX_WEIGHT = 1e-2    # link + entropy regularizers

# ============ Utils ============
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_undirected(edge_index, num_nodes):
    # Unique undirected edges without self loops
    edges = edge_index.t().tolist()
    uniq = set()
    out = []
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key not in uniq:
            uniq.add(key)
            out.append([a, b])
    if not out:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(out, dtype=torch.long).t().contiguous()

def macro_f1_from_logits(logits, y, mask):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        y_ = y[mask]
        p_ = pred[mask]
        C = int(y.max().item() + 1)
        cm = torch.zeros((C, C), dtype=torch.long, device=logits.device)
        for t, q in zip(y_, p_):
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

# ============ Data ============
def load_cora_from_content_and_cites(content_path: str, cites_path: str):
    lines = Path(content_path).read_text().strip().splitlines()
    n = len(lines)
    paper_ids, features, labels_raw = [], [], []
    for line in lines:
        toks = line.strip().split()
        paper_ids.append(toks[0])
        labels_raw.append(toks[-1])
        features.append([int(x) for x in toks[1:-1]])
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

# ============ L-RMC seeds and pooling ============
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
        miss = int((cluster_id < 0).sum().item())
        raise RuntimeError(f"{miss} nodes not covered by seeds.")
    K = int(cluster_id.max().item() + 1)
    return cluster_id, K

def pool_by_partition_weighted(x, edge_index, cluster_id, K):
    if x.dim() != 2:
        raise ValueError(f"Expected x to have shape [N, F], got {x.shape}")
    if cluster_id.shape != (x.shape[0],):
        raise ValueError(f"Expected cluster_id to have shape [{x.shape[0]}], got {cluster_id.shape}")
    sums = torch.zeros((K, x.size(1)), device=x.device, dtype=x.dtype)
    sums.index_add_(0, cluster_id, x)
    counts = torch.bincount(cluster_id, minlength=K).clamp_min(1).to(x.device).unsqueeze(1).to(x.dtype)
    x_pooled = sums / counts
    cu = cluster_id[edge_index[0]]
    cv = cluster_id[edge_index[1]]
    pairs = torch.stack([cu, cv], dim=1)
    uniq, w = torch.unique(pairs, dim=0, return_counts=True)
    mask = uniq[:, 0] != uniq[:, 1]
    edge_index_pooled = uniq[mask].t().contiguous()
    edge_weight = w[mask].to(torch.float)
    return x_pooled, edge_index_pooled, edge_weight

def compress_partition_to_K(cluster_id, K_target, edge_index):
    cid = cluster_id.clone()
    K_now = int(cid.max().item() + 1)
    if K_now <= K_target:
        return cid, K_now
    sizes = torch.bincount(cid, minlength=K_now)
    kept = set(int(k) for k in torch.topk(sizes, K_target).indices.tolist())
    # inter-cluster weights
    cu = cid[edge_index[0]].tolist()
    cv = cid[edge_index[1]].tolist()
    w = {}
    for a, b in zip(cu, cv):
        if a == b:
            continue
        w[(a, b)] = w.get((a, b), 0) + 1
        w[(b, a)] = w.get((b, a), 0) + 1
    mapping = {}
    largest_kept = max(kept, key=lambda k: sizes[k].item())
    for c in range(K_now):
        if c in kept:
            mapping[c] = c
        else:
            candidates = [(w.get((c, k), 0), k) for k in kept]
            mapping[c] = max(candidates)[1] if candidates else largest_kept
    for i in range(cid.numel()):
        cid[i] = mapping[int(cid[i].item())]
    kept_sorted = sorted(set(int(x) for x in cid.tolist()))
    remap = {old: new for new, old in enumerate(kept_sorted)}
    for i in range(cid.numel()):
        cid[i] = remap[int(cid[i].item())]
    return cid, len(kept_sorted)

# ============ Models ============
class LrmcSeededPoolGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, cluster_id, K, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        self.lin_skip = nn.Linear(hidden_dim, out_dim, bias=True)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = dropout
        self.register_buffer("cluster_id", cluster_id)
        self.K = K

    def forward(self, x, edge_index):
        if x.dim() != 2:
            raise ValueError(f"Expected x to have shape [N, F], got {x.shape}")
        x1 = F.relu(self.conv1(x, edge_index))
        if x1.shape[1] != HIDDEN:
            raise ValueError(f"Expected x1 to have shape [N, {HIDDEN}], got {x1.shape}")
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        gate = torch.tanh(self.score(x1))  # Remove .unsqueeze(-1)
        if gate.shape != (x1.shape[0], 1):
            raise ValueError(f"Expected gate to have shape [{x1.shape[0]}, 1], got {gate.shape}")
        x1_g = x1 * gate
        if x1_g.shape != x1.shape:
            raise ValueError(f"Expected x1_g to have shape {x1.shape}, got {x1_g.shape}")
        x_p, ei_p, ew_p = pool_by_partition_weighted(x1_g, edge_index, self.cluster_id, self.K)
        x_p = self.conv2(x_p, ei_p, edge_weight=ew_p)
        up = x_p[self.cluster_id]
        skip = self.lin_skip(x1)
        logits = up + skip
        return logits, 0.0

class TopKPoolBroadcastGCN(nn.Module):
    # gPool-style: learn scores, keep K, assign dropped to nearest kept by degree, weighted pooled GCN + skip.
    def __init__(self, in_dim, hidden_dim, out_dim, K_target, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        self.lin_skip = nn.Linear(hidden_dim, out_dim, bias=True)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = dropout
        self.K_target = K_target
    @staticmethod
    def _degrees(edge_index, N):
        return torch.bincount(edge_index[0], minlength=N).to(torch.long)
    def forward(self, x, edge_index):
        N = x.size(0)
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        raw = self.score(x1).squeeze(-1)
        gate = torch.tanh(raw).unsqueeze(-1)
        x1_g = x1 * gate
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
            if keep_mask[u]:
                continue
            cand = [w for w in neigh[u] if keep_mask[w]]
            cluster_id[u] = cluster_id[max(cand, key=lambda z: int(deg[z].item()))] if cand else cluster_id[best_global_kept]
        Kc = int(cluster_id.max().item() + 1)
        x_p, ei_p, ew_p = pool_by_partition_weighted(x1_g, edge_index, cluster_id, Kc)
        x_p = self.conv2(x_p, ei_p, edge_weight=ew_p)
        up = x_p[cluster_id]
        skip = self.lin_skip(x1)
        logits = up + skip
        return logits, 0.0

class DiffPoolGCNNode(nn.Module):
    # One DiffPool layer with K clusters, skip head to nodes.
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
        adj_dense = torch.zeros((N, N), device=device)
        adj_dense[edge_index[0], edge_index[1]] = 1.0
        idx = torch.arange(N, device=device)
        adj_dense[idx, idx] = 1.0
        x = x.unsqueeze(0)           # [1, N, F]
        adj = adj_dense.unsqueeze(0) # [1, N, N]
        mask = torch.ones((1, N), device=device)
        z = F.relu(self.gnn_embed1(x, adj, mask))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.relu(self.gnn_embed2(z, adj, mask))
        s = F.relu(self.gnn_assign1(x, adj, mask))
        s = F.dropout(s, p=self.dropout, training=self.training)
        s = self.gnn_assign2(s, adj, mask).softmax(dim=-1)  # [1, N, K]
        x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        h = F.relu(self.gnn_post1(x_pool, adj_pool))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gnn_post2(h, adj_pool)                     # [1, K, C]
        skip = self.lin_skip(z.squeeze(0))                  # [N, C]
        logits_nodes = torch.matmul(s.squeeze(0), h.squeeze(0)) + skip
        aux_loss = link_loss + ent_loss
        return logits_nodes, aux_loss

# ============ Train ============
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
        test_acc = accuracy_from_logits(logits, data.y, test_mask)
        test_f1  = macro_f1_from_logits(logits, data.y, test_mask)
    return test_acc, test_f1

# ============ Sweep runner ============
def run_sweeps():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_cora_from_content_and_cites(CORA_CONTENT, CORA_CITES)
    N = data.num_nodes
    cluster_id_full, K_full = load_lrmc_partition(SEEDS_JSON, data.num_nodes)

    print(f"Loaded Cora: N={data.num_nodes}, E={data.edge_index.size(1)}, F={data.num_features}, C={data.num_classes}")
    print(f"L-RMC base K = {K_full} (K/N = {K_full/N:.3f})")

    print("\nResults averaged over seeds:", SEEDS)
    print("tpc, K/N, K, Method, acc_mean, acc_std, f1_mean, f1_std")

    for tpc in LABEL_BUDGETS:
        for ratio in K_RATIOS:
            K_target = max(1, int(ratio * N))
            accs = { "LRMC": [], "gPool": [], "DiffPool": [] }
            f1s  = { "LRMC": [], "gPool": [], "DiffPool": [] }

            for s in SEEDS:
                set_seed(s)
                train_mask, val_mask, test_mask = make_planetoid_style_split(
                    data.y, data.num_classes, train_per_class=tpc, val_size=500, test_size=1000
                )

                # Equal K across methods
                cid_eq, K_eq = compress_partition_to_K(cluster_id_full, K_target, data.edge_index)

                # L-RMC
                lrmc_model = LrmcSeededPoolGCN(
                    in_dim=data.num_features, hidden_dim=HIDDEN, out_dim=data.num_classes,
                    cluster_id=cid_eq.to(data.x.device), K=K_eq, dropout=DROPOUT,
                )
                a, f = train_one(lrmc_model, data, train_mask, val_mask, test_mask, device)
                accs["LRMC"].append(a); f1s["LRMC"].append(f)

                # gPool
                g_model = TopKPoolBroadcastGCN(
                    in_dim=data.num_features, hidden_dim=HIDDEN, out_dim=data.num_classes,
                    K_target=K_eq, dropout=DROPOUT,
                )
                a, f = train_one(g_model, data, train_mask, val_mask, test_mask, device)
                accs["gPool"].append(a); f1s["gPool"].append(f)

                # DiffPool
                d_model = DiffPoolGCNNode(
                    in_dim=data.num_features, hidden_dim=HIDDEN, out_dim=data.num_classes,
                    K_clusters=K_eq, dropout=0.3,   # a little lower dropout helps DiffPool
                )
                a, f = train_one(d_model, data, train_mask, val_mask, test_mask, device,
                                 aux_weight=DIFFPOOL_AUX_WEIGHT)
                accs["DiffPool"].append(a); f1s["DiffPool"].append(f)

            def ms(x):  # mean, std
                return mean(x), (0.0 if len(x) < 2 else pstdev(x))

            for name in ["LRMC", "gPool", "DiffPool"]:
                am, asd = ms(accs[name])
                fm, fsd = ms(f1s[name])
                print(f"{tpc:3d}, {ratio:0.2f}, {K_eq:4d}, {name:7s}, "
                      f"{am:.3f}, {asd:.3f}, {fm:.3f}, {fsd:.3f}")

if __name__ == "__main__":
    run_sweeps()
