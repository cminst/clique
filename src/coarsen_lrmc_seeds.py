
# coarsen_lrmc_seeds.py
# Usage examples:
#   python coarsen_lrmc_seeds.py --seeds_json seeds.json --out_json seeds_K600.json --target_k 600
#   python coarsen_lrmc_seeds.py --seeds_json seeds.json --out_json seeds_ratio04.json --k_ratio 0.4
#
# This script:
#   1) loads Cora (Planetoid, normalized features)
#   2) reads your LRMC seeds JSON (clusters with "members")
#   3) computes a prototype (mean feature) per seed and its size
#   4) runs *weighted* k-means on the seed prototypes to coarsen to target_K
#   5) maps each node to its meta-cluster and writes a new seeds JSON
#
# It also prints cluster stats and the majority-vote upper bound before/after.

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
    K_guess = 0
    for c in obj["clusters"]:
        cid = int(c.get("cluster_id", K_guess))
        K_guess = max(K_guess, cid + 1)
        for u in c["members"]:
            cid_of_node[int(u)] = cid
    cluster_id = torch.full((num_nodes,), -1, dtype=torch.long)
    for u, cid in cid_of_node.items():
        if 0 <= u < num_nodes:
            cluster_id[u] = cid
    return cluster_id

def fix_uncovered_nodes(cluster_id: torch.Tensor) -> torch.Tensor:
    # Map uncovered nodes to a new single cluster (their own), so we retain all nodes.
    # Caller may later coarsen them away.
    N = cluster_id.numel()
    next_cid = int(cluster_id.max().item()) + 1 if (cluster_id >= 0).any() else 0
    for u in range(N):
        if cluster_id[u] < 0:
            cluster_id[u] = next_cid
            next_cid += 1
    return cluster_id

def prototypes_from_partition(X: torch.Tensor, cluster_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    K = int(cluster_id.max().item() + 1)
    F = X.size(1)
    device = X.device
    sums = torch.zeros(K, F, device=device, dtype=X.dtype)
    sizes = torch.bincount(cluster_id, minlength=K).to(device)
    sums.index_add_(0, cluster_id, X)
    sizes = sizes.clamp_min(1).to(X.dtype).unsqueeze(1)  # [K,1]
    protos = sums / sizes
    sizes = sizes.squeeze(1)
    return protos, sizes

def weighted_kmeans(protos: torch.Tensor, weights: torch.Tensor, target_K: int, iters: int = 30, seed: int = 0) -> torch.Tensor:
    """
    protos: [K0, F] seed prototypes
    weights: [K0] positive weights (e.g., cluster sizes)
    returns: [K0] meta-cluster id in [0, target_K)
    """
    torch.manual_seed(seed)
    K0, F = protos.shape
    target_K = min(target_K, K0)
    # init: k-means++-ish by weighted farthest point
    centers = torch.empty(target_K, F, device=protos.device, dtype=protos.dtype)
    chosen = torch.zeros(K0, dtype=torch.bool, device=protos.device)
    # pick first by weighted probability
    p0 = (weights / weights.sum()).clamp(min=1e-12)
    idx0 = torch.multinomial(p0, 1).item()
    centers[0] = protos[idx0]
    chosen[idx0] = True
    dist2 = (protos - centers[0:1]).pow(2).sum(dim=1)
    for k in range(1, target_K):
        # probability proportional to weight * distance^2 from nearest center
        prob = (weights * dist2).clamp(min=1e-12)
        prob = prob / prob.sum()
        idx = torch.multinomial(prob, 1).item()
        centers[k] = protos[idx]
        chosen[idx] = True
        dist2 = torch.minimum(dist2, (protos - centers[k:k+1]).pow(2).sum(dim=1))

    # Lloyd iterations (weighted)
    assign = torch.zeros(K0, dtype=torch.long, device=protos.device)
    for _ in range(iters):
        # assign
        d2 = (protos[:, None, :] - centers[None, :, :]).pow(2).sum(dim=2)  # [K0, target_K]
        assign = d2.argmin(dim=1)
        # update
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(target_K, device=protos.device, dtype=protos.dtype)
        new_centers.index_add_(0, assign, protos * weights.unsqueeze(1))
        counts.index_add_(0, assign, weights)
        mask = counts > 0
        new_centers[mask] = new_centers[mask] / counts[mask].unsqueeze(1).clamp_min(1e-12)
        # keep previous center where empty
        centers = torch.where(mask.unsqueeze(1), new_centers, centers)
    return assign

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

def cluster_size_stats(cluster_id: torch.Tensor) -> str:
    sizes = torch.bincount(cluster_id, minlength=int(cluster_id.max().item() + 1)).to(torch.float)
    singletons = (sizes == 1).float().mean().item()
    med = sizes.median().item()
    mean = sizes.mean().item()
    K = sizes.numel()
    return f"K={K}, singleton_rate={singletons:.3f}, mean_size={mean:.2f}, median_size={med:.2f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--target_k", type=int, default=None, help="Exact target number of clusters.")
    ap.add_argument("--k_ratio", type=float, default=None, help="Use target_k = ceil(k_ratio * N).")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, num_classes = load_cora(normalize=True)
    N = data.num_nodes

    cluster_id = read_seed_json(args.seeds_json, N)
    if (cluster_id < 0).any():
        print("[warn] Some nodes uncovered by seeds. Assigning unique temp clusters to uncovered nodes.")
        cluster_id = fix_uncovered_nodes(cluster_id)

    print("Before:", cluster_size_stats(cluster_id))
    ub_before = majority_vote_upper_bound(cluster_id, data.y)
    print(f"Majority-vote UB (before) = {ub_before:.3f}")

    # Determine target K
    if args.target_k is None and args.k_ratio is None:
        raise SystemExit("Provide either --target_k or --k_ratio.")
    target_K = int(args.target_k) if args.target_k is not None else int((args.k_ratio * N) + 0.999)
    # prototypes & weights
    X = data.x.to(torch.float)
    protos, sizes = prototypes_from_partition(X, cluster_id)
    K0 = protos.size(0)
    if target_K >= K0:
        print(f"[info] target_K ({target_K}) >= current K ({K0}); nothing to coarsen. Copying input to output.")
        out_cluster_id = cluster_id
    else:
        assign = weighted_kmeans(protos, sizes.clamp_min(1), target_K, iters=args.iters, seed=args.seed)  # [K0] seed -> meta
        out_cluster_id = assign[cluster_id]  # [N]

    print("After: ", cluster_size_stats(out_cluster_id))
    ub_after = majority_vote_upper_bound(out_cluster_id, data.y)
    print(f"Majority-vote UB (after)  = {ub_after:.3f}")

    # Write JSON
    K_final = int(out_cluster_id.max().item() + 1)
    clusters: List[Dict] = []
    for k in range(K_final):
        members = torch.nonzero(out_cluster_id == k, as_tuple=False).view(-1).tolist()
        clusters.append({"cluster_id": int(k), "members": members})
    out = {"clusters": clusters}
    Path(args.out_json).write_text(json.dumps(out))
    print(f"Wrote coarsened seeds to {args.out_json}")

if __name__ == "__main__":
    main()
