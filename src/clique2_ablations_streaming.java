import java.util.*;
import java.util.function.Consumer;

/**
 * clique2_ablations (streaming-capable)
 *
 * Provides two entry points:
 *  - runLaplacianRMC(List<Integer>[] adj1Based): materializes all snapshots (OK for small graphs)
 *  - runLaplacianRMCStreaming(List<Integer>[] adj1Based, Consumer<SnapshotDTO> sink): streams snapshots
 *
 * Implementation sketch:
 *  - Compute a degeneracy-style peeling order (min-degree removal) on the 0-1 adjacency.
 *  - Reconstruct in reverse: add nodes back; maintain DSU with per-component stats:
 *      * size (nC)
 *      * sumDegIn  = 2 * |E(C)|   (internal degree sum)
 *      * boundary  = # edges with exactly one endpoint in C   (we expose this as Q in SnapshotDTO)
 *      * members   = dynamic list of node ids in the component (merged by size)
 *  - After each addition step, emit a SnapshotDTO for the component that contains the added node.
 *
 * Notes:
 *  - Adjacency is expected 1-based: adj1Based[u1] lists v1 in 1..n. We convert to 0-based internally.
 *  - This code avoids Java Streams and IntStream.toArray to keep heap usage predictable on large graphs.
 */
public class clique2_ablations_streaming {

    // ---------------------------- API ----------------------------

    public static List<SnapshotDTO> runLaplacianRMC(List<Integer>[] adj1Based) {
        ArrayList<SnapshotDTO> out = new ArrayList<>();
        runLaplacianRMCStreaming(adj1Based, out::add);
        return out;
    }

    public static void runLaplacianRMCStreaming(List<Integer>[] adj1Based,
                                                Consumer<SnapshotDTO> sink) {
        final int n = adj1Based.length - 1; // 1-based
        // Build 0-based adjacency
        int[][] nbrs = new int[n][];
        int[] deg = new int[n];
        for (int u1 = 1; u1 <= n; u1++) {
            List<Integer> lst = adj1Based[u1];
            int m = lst.size();
            int[] arr = new int[m];
            for (int i = 0; i < m; i++) arr[i] = lst.get(i) - 1;
            nbrs[u1 - 1] = arr;
            deg[u1 - 1] = m;
        }

        // Peeling order (min-degree removal). Returns an array 'order' of length n,
        // where order[t] is the node removed at time t (0..n-1). We then add in reverse.
        int[] order = degeneracyOrder(nbrs, deg);

        // Reconstruction with DSU and per-component stats
        DSU dsu = new DSU(n);
        boolean[] added = new boolean[n];
        int[] tmpRoots = new int[16];
        int[] tmpCounts = new int[16];

        for (int step = n - 1; step >= 0; step--) {
            int u = order[step];
            added[u] = true;
            dsu.makeSingleton(u, deg[u]); // boundary starts at full degree; sumDegIn=0

            // Count u's neighbors by component root (among already-added nodes)
            int unique = 0;
            int[] Nu = nbrs[u];
            for (int v : Nu) {
                if (!added[v]) continue;
                int r = dsu.find(v);
                // linear search over small tmpRoots
                int idx = -1;
                for (int i = 0; i < unique; i++) if (tmpRoots[i] == r) { idx = i; break; }
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

            // Merge u's singleton with each neighbor component; track cross-edge counts
            int ru = dsu.find(u);
            for (int i = 0; i < unique; i++) {
                int rv = tmpRoots[i];
                if (ru == rv) continue;
                int t = tmpCounts[i]; // #edges between current set (ru) and component rv (all from u)
                ru = dsu.unionWithEdgeCount(ru, rv, t);
            }
            // After unions, ru is the root of u's component. We must also account for multiple
            // edges to the same component in case u had >1 neighbor in rv: handled via t above.
            // NOTE: sumDegIn increased by 2*sum t across neighbors; boundary updated accordingly.

            // Emit snapshot for the component containing u
            SnapshotDTO snap = dsu.snapshotOf(ru);
            sink.accept(snap);

            // reset tmp arrays
            for (int i = 0; i < unique; i++) {
                tmpRoots[i] = 0;
                tmpCounts[i] = 0;
            }
        }
    }

    // ------------------------- Degeneracy Order -------------------------

    // Computes a min-degree removal order without priority queue using bucket lists.
    static int[] degeneracyOrder(int[][] nbrs, int[] deg0) {
        final int n = nbrs.length;
        int maxDeg = 0;
        for (int d : deg0) if (d > maxDeg) maxDeg = d;

        int[] deg = Arrays.copyOf(deg0, n);
        int[] head = new int[maxDeg + 2]; // bucket heads
        int[] next = new int[n];
        int[] prev = new int[n];
        int[] whereDeg = new int[n]; // current bucket index per node

        Arrays.fill(head, -1);
        Arrays.fill(next, -1);
        Arrays.fill(prev, -1);

        for (int u = 0; u < n; u++) {
            int d = deg[u];
            whereDeg[u] = d;
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

            // decrease degrees of neighbors not yet removed
            for (int v : nbrs[u]) if (!removed[v]) {
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

    // --------------------------- DSU & Stats ---------------------------

    static final class DSU {
        final int n;
        final int[] parent;
        final int[] size;
        final long[] sumDegIn;  // internal degree sum (2*|E(C)|)
        final long[] boundary;  // edges with exactly one endpoint in C  (used as Q)
        final IntList[] members;

        DSU(int n) {
            this.n = n;
            this.parent = new int[n];
            this.size = new int[n];
            this.sumDegIn = new long[n];
            this.boundary = new long[n];
            this.members = new IntList[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                size[i] = 0;
                sumDegIn[i] = 0L;
                boundary[i] = 0L;
                members[i] = new IntList();
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
            // path compression
            int y = x;
            while (y != r) { int p = parent[y]; parent[y] = r; y = p; }
            return r;
        }

        // Merge sets 'ra' and 'rb' given t = number of edges between them (here, from the newly added node to rb)
        int unionWithEdgeCount(int ra, int rb, int t) {
            ra = find(ra); rb = find(rb);
            if (ra == rb) return ra;
            // union by size
            if (size[ra] < size[rb]) { int tmp = ra; ra = rb; rb = tmp; }
            // merge rb into ra
            parent[rb] = ra;
            // stats
            sumDegIn[ra] += sumDegIn[rb] + 2L * t;
            boundary[ra] += boundary[rb] - 2L * t;
            size[ra] += size[rb];
            // members: append smaller to larger
            members[ra].addAll(members[rb]);
            members[rb].clear();
            return ra;
        }

        SnapshotDTO snapshotOf(int r) {
            r = find(r);
            int sz = size[r];
            int[] nodes = members[r].toArray(); // single fresh array; caller may keep or discard
            return new SnapshotDTO(nodes, sz, sumDegIn[r], (double) boundary[r]);
        }
    }

    // Simple dynamic int list (no boxing)
    static final class IntList {
        int[] a;
        int sz;
        IntList() { a = new int[4]; sz = 0; }
        void clear() { sz = 0; }
        void add(int x) { if (sz == a.length) a = Arrays.copyOf(a, sz << 1); a[sz++] = x; }
        void addAll(IntList other) {
            if (other.sz == 0) return;
            int need = sz + other.sz;
            if (need > a.length) {
                int cap = a.length;
                while (cap < need) cap <<= 1;
                a = Arrays.copyOf(a, cap);
            }
            System.arraycopy(other.a, 0, a, sz, other.sz);
            sz = need;
        }
        int[] toArray() { return Arrays.copyOf(a, sz); }
    }

    // ------------------------- Snapshot DTO -------------------------

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
