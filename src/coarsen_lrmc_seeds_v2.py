
# coarsen_lrmc_seeds_v2.py
# Graph-aware coarsener for LRMC seeds.
#
# 1) Load Cora (normalized features) and your seeds JSON.
# 2) Build a *seed meta-graph* where edge weight w_ij = #edges between seed i and j.
# 3) Greedily MERGE tiny seeds (size < min_size) into the neighbor with the highest
#    combined score: score = lambda_conn * norm_conn + (1 - lambda_conn) * cosine.
#    - norm_conn = w_ij / sqrt(|Ci| * |Cj|)
#    - cosine = cos(proto_i, proto_j)
# 4) If K is still above target_K, finish with weighted k-means on seed prototypes.
#
# This cuts singleton rate dramatically *without* destroying label purity as much
# as feature-only k-means.
#
# Usage:
#   python coarsen_lrmc_seeds_v2.py --seeds_json seeds.json --out_json seeds_K700.json \
#       --target_k 700 --min_size 5 --lambda_conn 0.7 --finish_with_kmeans
#
#   python coarsen_lrmc_seeds_v2.py --seeds_json seeds.json --out_json seeds_ratio025.json \
#       --k_ratio 0.25 --min_size 4 --lambda_conn 0.6 --finish_with_kmeans
#
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

def load_cora(normalize=True):
    ds = Planetoid(root="/tmp/Cora", name="Cora", transform=T.NormalizeFeatures() if normalize else None)
    return ds[0], ds.num_classes

def read_seed_json(path: str, num_nodes: int) -> torch.Tensor:
    obj = json.loads(Path(path).read_text())
    cid_of_node: Dict[int, int] = {}
    next_id = 0
    for c in obj["clusters"]:
        cid = int(c.get("cluster_id", next_id))
        next_id = max(next_id, cid + 1)
        for u in c["seed_nodes"]:
            cid_of_node[int(u)] = cid
    cluster_id = torch.full((num_nodes,), -1, dtype=torch.long)
    for u, cid in cid_of_node.items():
        if 0 <= u < num_nodes:
            cluster_id[u] = cid
    # Reindex
    uniq = torch.unique(cluster_id[cluster_id >= 0]).tolist()
    remap = {int(old): i for i, old in enumerate(uniq)}
    for u in range(num_nodes):
        if cluster_id[u] >= 0:
            cluster_id[u] = remap[int(cluster_id[u].item())]
    return cluster_id

def fix_uncovered_nodes(cluster_id: torch.Tensor) -> torch.Tensor:
    N = cluster_id.numel()
    next_cid = int(cluster_id.max().item()) + 1 if (cluster_id >= 0).any() else 0
    for u in range(N):
        if cluster_id[u] < 0:
            cluster_id[u] = next_cid
            next_cid += 1
    return cluster_id

def cluster_size_stats(cluster_id: torch.Tensor) -> str:
    sizes = torch.bincount(cluster_id, minlength=int(cluster_id.max().item() + 1)).to(torch.float)
    singletons = (sizes == 1).float().mean().item()
    med = sizes.median().item()
    mean = sizes.mean().item()
    K = sizes.numel()
    return f"K={K}, singleton_rate={singletons:.3f}, mean_size={mean:.2f}, median_size={med:.2f}"

def majority_vote_upper_bound(cluster_id: torch.Tensor, y: torch.Tensor) -> float:
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

def prototypes_from_partition(X: torch.Tensor, cluster_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    K = int(cluster_id.max().item() + 1)
    F = X.size(1)
    device = X.device
    sums = torch.zeros(K, F, device=device, dtype=X.dtype)
    sizes = torch.bincount(cluster_id, minlength=K).to(device)
    sums.index_add_(0, cluster_id, X)
    sizes = sizes.clamp_min(1).to(X.dtype).unsqueeze(1)
    protos = sums / sizes
    sizes = sizes.squeeze(1)
    return protos, sizes

def build_seed_metagraph(edge_index: torch.Tensor, cluster_id: torch.Tensor, K: int) -> Dict[tuple, int]:
    u = cluster_id[edge_index[0]]
    v = cluster_id[edge_index[1]]
    pairs = torch.stack([u, v], dim=1)
    mask = pairs[:,0] != pairs[:,1]
    pairs = pairs[mask]
    d: Dict[tuple, int] = {}
    for a, b in pairs.tolist():
        if a > b: a, b = b, a
        d[(a, b)] = d.get((a, b), 0) + 1
    return d

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), eps=1e-12).item())

def graph_aware_merge(data, cluster_id: torch.Tensor, min_size: int, lambda_conn: float, max_iters: int = 8) -> torch.Tensor:
    X = data.x.to(torch.float)
    device = X.device
    for _ in range(max_iters):
        protos, sizes = prototypes_from_partition(X, cluster_id)
        K = int(cluster_id.max().item() + 1)
        meta = build_seed_metagraph(data.edge_index.to(device), cluster_id, K)
        neigh = [[] for _ in range(K)]
        for (a,b), w in meta.items():
            neigh[a].append((b, w))
            neigh[b].append((a, w))

        sizes_list = torch.bincount(cluster_id, minlength=K).tolist()
        tiny = [k for k, s in enumerate(sizes_list) if s < min_size]
        if not tiny:
            break

        parent = list(range(K))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        merges = []
        for a in tiny:
            cand = neigh[a]
            best_score = -1.0
            best_b = None
            for b, w in cand:
                na, nb = sizes_list[a], sizes_list[b]
                norm_conn = w / ((na * nb) ** 0.5 + 1e-12)
                cos = cosine_sim(protos[a], protos[b])
                score = lambda_conn * norm_conn + (1 - lambda_conn) * max(cos, 0.0)
                if score > best_score:
                    best_score, best_b = score, b
            if best_b is None:
                cos_all = torch.matmul(F.normalize(protos, dim=1), F.normalize(protos[a:a+1], dim=1).t()).squeeze(1)
                cos_all[a] = -1.0
                best_b = int(torch.argmax(cos_all).item())
            merges.append((a, best_b))

        for a, b in merges:
            ra, rb = find(a), find(b)
            if ra != rb:
                union(ra, rb)

        root_map = {}
        new_id = 0
        for k in range(K):
            r = find(k)
            if r not in root_map:
                root_map[r] = new_id
                new_id += 1

        lut = torch.tensor([root_map[find(k)] for k in range(K)], dtype=torch.long, device=cluster_id.device)
        new_cluster_id = lut[cluster_id]
        if int(new_cluster_id.max().item()) + 1 == K:
            break
        cluster_id = new_cluster_id

    return cluster_id

def weighted_kmeans(protos: torch.Tensor, weights: torch.Tensor, target_K: int, iters: int = 30, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    K0, F = protos.shape
    target_K = min(target_K, K0)
    centers = torch.empty(target_K, F, device=protos.device, dtype=protos.dtype)
    p0 = (weights / weights.sum()).clamp(min=1e-12)
    idx0 = torch.multinomial(p0, 1).item()
    centers[0] = protos[idx0]
    dist2 = (protos - centers[0:1]).pow(2).sum(dim=1)
    for k in range(1, target_K):
        prob = (weights * dist2).clamp(min=1e-12)
        prob = prob / prob.sum()
        idx = torch.multinomial(prob, 1).item()
        centers[k] = protos[idx]
        dist2 = torch.minimum(dist2, (protos - centers[k:k+1]).pow(2).sum(dim=1))

    assign = torch.zeros(K0, dtype=torch.long, device=protos.device)
    for _ in range(iters):
        d2 = (protos[:, None, :] - centers[None, :, :]).pow(2).sum(dim=2)
        assign = d2.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(target_K, device=protos.device, dtype=protos.dtype)
        new_centers.index_add_(0, assign, protos * weights.unsqueeze(1))
        counts.index_add_(0, assign, weights)
        mask = counts > 0
        new_centers[mask] = new_centers[mask] / counts[mask].unsqueeze(1).clamp_min(1e-12)
        centers = torch.where(mask.unsqueeze(1), new_centers, centers)
    return assign

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--target_k", type=int, default=None)
    ap.add_argument("--k_ratio", type=float, default=None)
    ap.add_argument("--min_size", type=int, default=5)
    ap.add_argument("--lambda_conn", type=float, default=0.7)
    ap.add_argument("--finish_with_kmeans", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, _ = load_cora(normalize=True)
    N = data.num_nodes

    cluster_id = read_seed_json(args.seeds_json, N)
    if (cluster_id < 0).any():
        print("[warn] Some nodes uncovered by seeds. Assigning temporary singletons.")
        cluster_id = fix_uncovered_nodes(cluster_id)

    print("== BEFORE ==")
    print(cluster_size_stats(cluster_id))
    print(f"Majority-vote UB (before) = {majority_vote_upper_bound(cluster_id, data.y):.3f}")

    # Stage 1
    cluster_id = graph_aware_merge(data, cluster_id, min_size=args.min_size, lambda_conn=args.lambda_conn, max_iters=8)

    print("\n== AFTER GRAPH-AWARE MERGE ==")
    print(cluster_size_stats(cluster_id))
    print(f"Majority-vote UB (stage1)  = {majority_vote_upper_bound(cluster_id, data.y):.3f}")

    # Target K
    target_K = None
    if args.target_k is not None:
        target_K = int(args.target_k)
    elif args.k_ratio is not None:
        target_K = int(args.k_ratio * N + 0.999)

    # Stage 2
    if target_K is not None:
        K_current = int(cluster_id.max().item() + 1)
        if target_K < K_current and args.finish_with_kmeans:
            X = data.x.to(torch.float)
            protos, sizes = prototypes_from_partition(X, cluster_id)
            assign = weighted_kmeans(protos, sizes.clamp_min(1), target_K, iters=args.iters, seed=args.seed)
            cluster_id = assign[cluster_id]

            print("\n== AFTER K-MEANS FINISH ==")
            print(cluster_size_stats(cluster_id))
            print(f"Majority-vote UB (final)  = {majority_vote_upper_bound(cluster_id, data.y):.3f}")
        else:
            print("\n[info] Skipping k-means finish (either K already <= target or --finish_with_kmeans not set).")

    # Write JSON
    K_final = int(cluster_id.max().item() + 1)
    clusters: List[Dict] = []
    for k in range(K_final):
        seed_nodes = torch.nonzero(cluster_id == k, as_tuple=False).view(-1).tolist()
        clusters.append({"cluster_id": int(k), "seed_nodes": seed_nodes})
    out = {"clusters": clusters}
    Path(args.out_json).write_text(json.dumps(out))
    print(f"\nWrote coarsened seeds to {args.out_json}")

if __name__ == "__main__":
    main()
