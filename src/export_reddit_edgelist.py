#!/usr/bin/env python3
# export_reddit_edgelist_canonical.py
# Writes EACH undirected edge exactly once: "u v" with u < v (0-based), from PyG Reddit.
# This halves the edge count relative to to_undirected and avoids duplication downstream.
#
# Usage:
#   python export_reddit_edgelist_canonical.py --out reddit_edges.txt --root ./data/Reddit

import argparse
from pathlib import Path
import torch
from torch_geometric.datasets import Reddit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./data/Reddit")
    ap.add_argument("--out", type=str, default="reddit_edges.txt")
    args = ap.parse_args()

    ds = Reddit(root=args.root); data = ds[0]
    ei = data.edge_index  # directed; in this dataset it's effectively undirected
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)

    # canonical pairs u<v; de-duplicate
    seen = set()
    with outp.open("w") as f:
        E = ei.size(1)
        for e in range(E):
            u = int(ei[0, e]); v = int(ei[1, e])
            if u == v:
                continue
            if u > v:
                u, v = v, u
            key = (u << 32) | v
            if key in seen:
                continue
            seen.add(key)
            f.write(f"{u} {v}\n")
    print(f"Wrote {len(seen)} undirected edges to {outp} (nodes: {data.num_nodes})")

if __name__ == "__main__":
    main()
