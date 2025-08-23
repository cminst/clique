# Generate large synthetic graph testcases in the requested format.
# We create ring (k-nearest-neighbor) graphs and a star graph.
# Format:
#   First line: "N M"
#   Each subsequent line: "u v" with 1-based node ids.
#
# Files created:
# - /mnt/data/ring_n10000_d8.txt
# - /mnt/data/ring_n100000_d8.txt
# - /mnt/data/ring_n1000000_d4.txt
# - /mnt/data/star_n1000000.txt
#
# The ring graphs guarantee exactly M = N * (d/2) undirected edges with no duplicates.
# The star graph has M = N - 1 edges.

import os
from time import time

def write_ring_graph(path: str, n: int, d: int) -> None:
    assert d % 2 == 0 and d >= 2
    m = n * (d // 2)
    t0 = time()
    with open(path, "w") as f:
        f.write(f"{n} {m}\n")
        buf = []
        # Flush every 100k edges to keep memory small and writes efficient
        FLUSH = 100_000
        written = 0
        for i in range(1, n + 1):
            # connect to next d/2 neighbors (wrap around)
            half = d // 2
            for off in range(1, half + 1):
                j = i + off
                if j > n:
                    j -= n
                buf.append(f"{i} {j}\n")
                written += 1
                if written % FLUSH == 0:
                    f.writelines(buf)
                    buf.clear()
        if buf:
            f.writelines(buf)
    t1 = time()
    print(f"Wrote ring graph {os.path.basename(path)}: N={n}, d={d}, M={m}, time={(t1 - t0):.2f}s")

def write_star_graph(path: str, n: int) -> None:
    m = n - 1
    t0 = time()
    with open(path, "w") as f:
        f.write(f"{n} {m}\n")
        buf = []
        FLUSH = 200_000
        for i in range(2, n + 1):
            buf.append(f"1 {i}\n")
            if len(buf) >= FLUSH:
                f.writelines(buf)
                buf.clear()
        if buf:
            f.writelines(buf)
    t1 = time()
    print(f"Wrote star graph {os.path.basename(path)}: N={n}, M={m}, time={(t1 - t0):.2f}s")

# Paths
ring_10k = "ring_n10000_d8.txt"
ring_100k = "ring_n100000_d8.txt"
ring_1m = "ring_n1000000_d4.txt"
star_1m = "star_n1000000.txt"

# Generate files
write_ring_graph(ring_10k, 10_000, 8)       # M = 40,000
write_ring_graph(ring_100k, 100_000, 8)     # M = 400,000
write_ring_graph(ring_1m, 1_000_000, 4)     # M = 2,000,000
write_star_graph(star_1m, 1_000_000)

# Show file sizes (MB)
for p in [ring_10k, ring_100k, ring_1m, star_1m]:
    size_mb = os.path.getsize(p) / (1024 * 1024)
    print(f"{os.path.basename(p)} -> {size_mb:.2f} MB")
