import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;
import java.util.stream.IntStream;

/**
 * clique2_ablations (streaming-capable, optimized)
 *
 * Optimizations:
 * - Parallel degeneracy computation where safe
 * - Better memory allocation strategies
 * - Optimized data structures
 * - Reduced object allocation in hot paths
 */
public class clique2_ablations_streaming {

    // API

    public static List<SnapshotDTO> runLaplacianRMC(List<Integer>[] adj1Based) {
        ArrayList<SnapshotDTO> out = new ArrayList<>();
        runLaplacianRMCStreaming(adj1Based, out::add);
        return out;
    }

    public static void runLaplacianRMCStreaming(List<Integer>[] adj1Based,
                                                Consumer<SnapshotDTO> sink) {
        final int n = adj1Based.length - 1; // 1-based
        System.out.printf("# Building 0-based adjacency for n=%d nodes...%n", n);

        // Build 0-based adjacency with optimized allocation
        int[][] nbrs = new int[n][];
        int[] deg = new int[n];

        // Parallel conversion from 1-based to 0-based
        IntStream.range(0, n).parallel().forEach(u0 -> {
            int u1 = u0 + 1;
            List<Integer> lst = adj1Based[u1];
            int m = lst.size();
            int[] arr = new int[m];
            for (int i = 0; i < m; i++) {
                arr[i] = lst.get(i) - 1;
            }
            nbrs[u0] = arr;
            deg[u0] = m;
        });

        System.out.println("# Computing degeneracy order...");
        long startTime = System.currentTimeMillis();

        // Peeling order (optimized degeneracy computation)
        int[] order = degeneracyOrderOptimized(nbrs, deg);

        long degTime = System.currentTimeMillis() - startTime;
        System.out.printf("# Degeneracy order computed in %.2f seconds%n", degTime / 1000.0);

        System.out.println("# Starting reconstruction phase...");
        startTime = System.currentTimeMillis();

        // Reconstruction with optimized DSU
        DSU dsu = new DSU(n);
        boolean[] added = new boolean[n];
        int[] tmpRoots = new int[32]; // Larger initial size
        int[] tmpCounts = new int[32];

        for (int step = n - 1; step >= 0; step--) {
            System.out.println("# Step " + step);
            int u = order[step];
            added[u] = true;
            dsu.makeSingleton(u, deg[u]); // boundary starts at full degree; sumDegIn=0

            // Count u's neighbors by component root (among already-added nodes)
            int unique = 0;
            int[] Nu = nbrs[u];
            for (int v : Nu) {
                if (!added[v]) continue;
                int r = dsu.find(v);
                // Optimized linear search with early termination
                int idx = -1;
                for (int i = 0; i < unique; i++) {
                    if (tmpRoots[i] == r) { idx = i; break; }
                }
                if (idx >= 0) {
                    tmpCounts[idx] += 1;
                } else {
                    if (unique == tmpRoots.length) {
                        tmpRoots = Arrays.copyOf(tmpRoots, unique * 2);
                        tmpCounts = Arrays.copyOf(tmpCounts, unique * 2);
                    }
                    tmpRoots[unique] = r;
                    tmpCounts[unique] = 1;
                    unique++;
                }
            }

            // Merge u's singleton with each neighbor component
            int ru = dsu.find(u);
            for (int i = 0; i < unique; i++) {
                int rv = tmpRoots[i];
                if (ru == rv) continue;
                int t = tmpCounts[i];
                ru = dsu.unionWithEdgeCount(ru, rv, t);
            }

            // Emit snapshot for the component containing u
            SnapshotDTO snap = dsu.snapshotOf(ru);
            sink.accept(snap);

            // Clear tmp arrays more efficiently
            Arrays.fill(tmpRoots, 0, unique, 0);
            Arrays.fill(tmpCounts, 0, unique, 0);
        }

        long reconTime = System.currentTimeMillis() - startTime;
        System.out.printf("# Reconstruction completed in %.2f seconds%n", reconTime / 1000.0);
    }

    // Optimized Degeneracy Order
    static int[] degeneracyOrderOptimized(int[][] nbrs, int[] deg0) {
        final int n = nbrs.length;
        int maxDeg = Arrays.stream(deg0).parallel().max().orElse(0);

        int[] deg = Arrays.copyOf(deg0, n);
        int[] head = new int[maxDeg + 2]; // bucket heads
        int[] next = new int[n];
        int[] prev = new int[n];

        Arrays.fill(head, -1);
        Arrays.fill(next, -1);
        Arrays.fill(prev, -1);

        // Initialize buckets
        for (int u = 0; u < n; u++) {
            int d = deg[u];
            // push front into bucket d
            next[u] = head[d];
            if (head[d] >= 0) prev[head[d]] = u;
            head[d] = u;
        }

        int[] order = new int[n];
        int ptr = 0;
        int cur = 0;
        while (cur <= maxDeg && head[cur] < 0) cur++;

        boolean[] removed = new boolean[n];

        for (int iter = 0; iter < n; iter++) {
            while (cur <= maxDeg && head[cur] < 0) cur++;
            if (cur > maxDeg) cur = maxDeg; // safety

            int u = head[cur];
            // pop u from bucket cur
            head[cur] = next[u];
            if (head[cur] >= 0) prev[head[cur]] = -1;
            next[u] = prev[u] = -1;

            removed[u] = true;
            order[ptr++] = u;

            // Optimized neighbor degree updates
            int[] neighbors = nbrs[u];
            for (int i = 0; i < neighbors.length; i++) {
                int v = neighbors[i];
                if (removed[v]) continue;

                int dv = deg[v];
                // remove v from bucket dv
                int pv = prev[v], nv = next[v];
                if (pv >= 0) next[pv] = nv; else head[dv] = nv;
                if (nv >= 0) prev[nv] = pv;

                // insert v into bucket dv-1
                int nd = dv - 1;
                deg[v] = nd;
                next[v] = head[nd];
                if (head[nd] >= 0) prev[head[nd]] = v;
                prev[v] = -1;
                head[nd] = v;
                if (nd < cur) cur = nd;
            }
        }
        return order;
    }

    // Optimized DSU & Stats
    static final class DSU {
        final int n;
        final int[] parent;
        final int[] size;
        final long[] sumDegIn;  // internal degree sum (2*|E(C)|)
        final long[] boundary;  // edges with exactly one endpoint in C
        final OptimizedIntList[] members;

        DSU(int n) {
            this.n = n;
            this.parent = new int[n];
            this.size = new int[n];
            this.sumDegIn = new long[n];
            this.boundary = new long[n];
            this.members = new OptimizedIntList[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                size[i] = 0;
                sumDegIn[i] = 0L;
                boundary[i] = 0L;
                members[i] = new OptimizedIntList();
            }
        }

        void makeSingleton(int u, int degU) {
            parent[u] = u;
            size[u] = 1;
            sumDegIn[u] = 0L;
            boundary[u] = degU;
            members[u].clear();
            members[u].add(u);
        }

        int find(int x) {
            int r = x;
            while (r != parent[r]) r = parent[r];
            // Optimized path compression
            while (x != r) {
                int p = parent[x];
                parent[x] = r;
                x = p;
            }
            return r;
        }

        int unionWithEdgeCount(int ra, int rb, int t) {
            ra = find(ra); rb = find(rb);
            if (ra == rb) return ra;

            // union by size
            if (size[ra] < size[rb]) {
                int tmp = ra; ra = rb; rb = tmp;
            }

            // merge rb into ra
            parent[rb] = ra;

            // stats
            sumDegIn[ra] += sumDegIn[rb] + 2L * t;
            boundary[ra] += boundary[rb] - 2L * t;
            size[ra] += size[rb];

            // members: optimized append
            members[ra].addAll(members[rb]);
            members[rb].clear();

            return ra;
        }

        SnapshotDTO snapshotOf(int r) {
            r = find(r);
            int sz = size[r];
            int[] nodes = members[r].toArray();
            return new SnapshotDTO(nodes, sz, sumDegIn[r], (double) boundary[r]);
        }
    }

    // Optimized dynamic int list with better memory management
    static final class OptimizedIntList {
        private int[] a;
        private int sz;

        OptimizedIntList() {
            a = new int[8]; // Start with larger initial capacity
            sz = 0;
        }

        void clear() { sz = 0; }

        void add(int x) {
            if (sz == a.length) {
                a = Arrays.copyOf(a, sz << 1); // Double capacity
            }
            a[sz++] = x;
        }

        void addAll(OptimizedIntList other) {
            if (other.sz == 0) return;
            int need = sz + other.sz;
            if (need > a.length) {
                int cap = Math.max(a.length << 1, need);
                a = Arrays.copyOf(a, cap);
            }
            System.arraycopy(other.a, 0, a, sz, other.sz);
            sz = need;
        }

        int[] toArray() {
            return Arrays.copyOf(a, sz);
        }
    }

    // Snapshot DTO

    public static final class SnapshotDTO {
        public final int[] nodes;   // 0-based ids in the original graph
        public final int nC;        // |C|
        public final long sumDegIn; // 2 * |E(C)|
        public final double Q;      // here: boundary(C) (edges leaving C)

        public SnapshotDTO(int[] nodes, int nC, long sumDegIn, double Q) {
            this.nodes = nodes;
            this.nC = nC;
            this.sumDegIn = sumDegIn;
            this.Q = Q;
        }
    }
}
