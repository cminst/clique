import sys

def reindex(in_path: str, out_path: str):
    with open(in_path, 'r') as f:
        header = f.readline().strip().split()
        edges = []
        nodes = set()
        for line in f:
            if not line.strip():
                continue
            a, b = map(int, line.split())
            edges.append((a, b))
            nodes.add(a); nodes.add(b)

    nodes_sorted = sorted(nodes)
    remap = {old: i+1 for i, old in enumerate(nodes_sorted)}
    n = len(nodes_sorted)
    m = len(edges)

    with open(out_path, 'w') as f:
        f.write(f"{n} {m}\n")
        for a, b in edges:
            f.write(f"{remap[a]} {remap[b]}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python reindex_graph_file.py <input> [output]", file=sys.stderr)
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) == 3 else inp.replace('.txt', '_reindexed.txt')
    reindex(inp, out)
    print(f"Wrote: {out}")

