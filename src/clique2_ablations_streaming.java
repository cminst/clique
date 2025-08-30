import java.util.*;
import java.util.function.Consumer;

/**
 * clique2_ablations (EXACT-Q, streaming-capable, fast)
 *
 * Entry points:
 *  - runLaplacianRMC(List<Integer>[] adj1Based): collects snapshots (OK for small graphs)
 *  - runLaplacianRMCStreaming(List<Integer>[] adj1Based, Consumer<SnapshotDTO> sink): streams snapshots
 *
 * Exact Q for ALL graph sizes:
 *   For a component C (induced subgraph), with full-graph degrees d_i:
 *     Q(C) = d^T L_C d = sum_i [deg_C(i) * d_i^2] - 2 * sum_{(i,j) in E_C, i<j} d_i d_j
 *
 * We maintain these per-component stats incrementally in DSU during reverse reconstruction:
 *   - sumDegIn2  = 2 * |E_C|
 *   - sumD2degC  = Σ_i deg_C(i) * d_i^2
 *   - sumEprod   = Σ_{(i,j)∈E_C, i<j} d_i * d_j
 * Then  Q = sumD2degC - 2 * sumEprod  (exact).
 *
 * Cost per step = O(deg(u)) + near-constant DSU unions. No Java streams or giant arrays.
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
        final int n = adj1Based.length - 1; // 1-based adjacency
        // Build 0-based adjacency + full degrees
        int[][] nbrs = new int[n][];
        int[] degFull = new int[n];
        for (int u1 = 1; u1 <= n; u1++) {
            List<Integer> lst = adj1Based[u1];
            int m = lst.size();
            int[] arr = new int[m];
            for (int i = 0; i < m; i++) arr[i] = lst.get(i) - 1;
            nbrs[u1 - 1] = arr;
            degFull[u1 - 1] = m;
        }

        // Peeling order: stable PQ for small graphs, bucket for large graphs
        int[] order = (n <= 100_000) ? degeneracyOrderStable(nbrs, degFull)
                                     : degeneracyOrderBucket(nbrs, degFull);

        DSU dsu = new DSU(n, degFull);
        boolean[] added = new boolean[n];

        // temp aggregators for neighbor components
        int[] tmpRoots = new int[8];
        int[] tmpCounts = new int[8];
        long[] tmpSumD  = new long[8];   // Σ d_v over neighbors in that component
        long[] tmpSumD2 = new long[8];   // Σ d_v^2 over neighbors in that component

        for (int step = n - 1; step >= 0; step--) {
            final int u = order[step];
            final int du = degFull[u];
            added[u] = true;
            dsu.makeSingleton(u);

            int unique = 0;
            int liveNbrs = 0;
            long totalSumD = 0L;
            long totalSumD2 = 0L;

            // Scan neighbors to aggregate per-component counts/sums
            for (int v : nbrs[u]) {
                if (!added[v]) continue;
                liveNbrs++;
                int r = dsu.find(v);
                int idx = -1;
                for (int i = 0; i < unique; i++) if (tmpRoots[i] == r) { idx = i; break; }
                if (idx < 0) {
                    if (unique == tmpRoots.length) {
                        int newLen = unique << 1;
                        tmpRoots = Arrays.copyOf(tmpRoots, newLen);
                        tmpCounts = Arrays.copyOf(tmpCounts, newLen);
                        tmpSumD = Arrays.copyOf(tmpSumD, newLen);
                        tmpSumD2 = Arrays.copyOf(tmpSumD2, newLen);
                    }
                    idx = unique++;
                    tmpRoots[idx] = r;
                    tmpCounts[idx] = 1;
                    tmpSumD[idx] = degFull[v];
                    tmpSumD2[idx] = (long) degFull[v] * (long) degFull[v];
                } else {
                    tmpCounts[idx] += 1;
                    tmpSumD[idx] += degFull[v];
                    tmpSumD2[idx] += (long) degFull[v] * (long) degFull[v];
                }
                totalSumD += degFull[v];
                totalSumD2 += (long) degFull[v] * (long) degFull[v];
            }

            // Merge u's singleton with each neighbor component (via DSU)
            int ru = dsu.find(u);
            for (int i = 0; i < unique; i++) {
                int rv = tmpRoots[i];
                if (ru == rv) continue;
                ru = dsu.union(ru, rv);
            }

            // Update per-component stats for edges formed by u
            // Internal edges added this step: t = liveNbrs, contributing:
            //   sumDegIn2 += 2 * t
            //   sumEprod  += du * Σ d_v  (v = added neighbors)
            //   sumD2degC += t * du^2 + Σ d_v^2
            long delta2 = 2L * liveNbrs;
            double deltaEprod = (double) du * (double) totalSumD;
            double deltaD2degC = (double) liveNbrs * (double) du * (double) du + (double) totalSumD2;
            dsu.addStats(ru, delta2, deltaEprod, deltaD2degC);

            // Emit snapshot for the component containing u
            int[] nodes = dsu.nodesOf(ru);
            long sumDegIn2 = dsu.sumDegIn2(ru);
            double Q = dsu.sumD2degC(ru) - 2.0 * dsu.sumEprod(ru);
            sink.accept(new SnapshotDTO(nodes, nodes.length, sumDegIn2, Q));

            // reset temp arrays entries we used
            for (int i = 0; i < unique; i++) {
                tmpRoots[i] = tmpCounts[i] = 0;
                tmpSumD[i] = tmpSumD2[i] = 0L;
            }
        }
    }

    // ------------------------- Degeneracy Order -------------------------

    // Stable min-degree removal via lazy PQ (ties by node id). O(E log N), good for small graphs.
    static int[] degeneracyOrderStable(int[][] nbrs, int[] deg0) {
        final int n = nbrs.length;
        int[] deg = Arrays.copyOf(deg0, n);
        boolean[] removed = new boolean[n];
        PriorityQueue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> (a[0] << 20) | a[1]));
        for (int u = 0; u < n; u++) pq.add(new long[]{deg[u], u});
        int[] order = new int[n];
        int ptr = 0;
        while (ptr < n) {
            long[] top = pq.poll();
            int du = (int) top[0];
            int u = (int) top[1];
            if (removed[u] || du != deg[u]) continue;
            removed[u] = true;
            order[ptr++] = u;
            for (int v : nbrs[u]) if (!removed[v]) {
                deg[v]--;
                pq.add(new long[]{deg[v], v});
            }
        }
        return order;
    }

    // Fast bucket-based degeneracy (no tie stability). O(E).
    static int[] degeneracyOrderBucket(int[][] nbrs, int[] deg0) {
        final int n = nbrs.length;
        int maxDeg = 0;
        for (int d : deg0) if (d > maxDeg) maxDeg = d;

        int[] deg = Arrays.copyOf(deg0, n);
        int[] head = new int[maxDeg + 2];
        int[] next = new int[n];
        int[] prev = new int[n];
        Arrays.fill(head, -1);
        Arrays.fill(next, -1);
        Arrays.fill(prev, -1);
        for (int u = 0; u < n; u++) {
            int d = deg[u];
            next[u] = head[d];
            if (head[d] >= 0) prev[head[d]] = u;
            head[d] = u;
        }

        boolean[] removed = new boolean[n];
        int[] order = new int[n];
        int ptr = 0;
        int cur = 0;
        while (cur <= maxDeg && head[cur] < 0) cur++;

        for (int it = 0; it < n; it++) {
            while (cur <= maxDeg && head[cur] < 0) cur++;
            if (cur > maxDeg) cur = maxDeg;
            int u = head[cur];
            head[cur] = next[u];
            if (head[cur] >= 0) prev[head[cur]] = -1;
            next[u] = prev[u] = -1;
            removed[u] = true;
            order[ptr++] = u;

            for (int v : nbrs[u]) if (!removed[v]) {
                int dv = deg[v];
                int pv = prev[v], nv = next[v];
                if (pv >= 0) next[pv] = nv; else head[dv] = nv;
                if (nv >= 0) prev[nv] = pv;
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
        final long[] sumDegIn2;    // 2 * |E_C|
        final double[] sumEprod;   // Σ d_i d_j over internal edges (i<j)
        final double[] sumD2degC;  // Σ deg_C(i) * d_i^2
        final int[] degFull;
        final IntList[] members;

        DSU(int n, int[] degFull) {
            this.n = n;
            this.parent = new int[n];
            this.size = new int[n];
            this.sumDegIn2 = new long[n];
            this.sumEprod = new double[n];
            this.sumD2degC = new double[n];
            this.degFull = degFull;
            this.members = new IntList[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                size[i] = 0;
                sumDegIn2[i] = 0L;
                sumEprod[i] = 0.0;
                sumD2degC[i] = 0.0;
                members[i] = new IntList();
            }
        }

        void makeSingleton(int u) {
            parent[u] = u;
            size[u] = 1;
            sumDegIn2[u] = 0L;
            sumEprod[u] = 0.0;
            sumD2degC[u] = 0.0;
            members[u].clear();
            members[u].add(u);
        }

        int find(int x) {
            int r = x;
            while (r != parent[r]) r = parent[r];
            int y = x;
            while (y != r) { int p = parent[y]; parent[y] = r; y = p; }
            return r;
        }

        int union(int ra, int rb) {
            ra = find(ra); rb = find(rb);
            if (ra == rb) return ra;
            if (size[ra] < size[rb]) { int t = ra; ra = rb; rb = t; }
            parent[rb] = ra;
            size[ra] += size[rb];
            sumDegIn2[ra] += sumDegIn2[rb];
            sumEprod[ra]  += sumEprod[rb];
            sumD2degC[ra] += sumD2degC[rb];
            members[ra].addAll(members[rb]);
            members[rb].clear();
            return ra;
        }

        void addStats(int r, long delta2, double deltaEprod, double deltaD2degC) {
            r = find(r);
            sumDegIn2[r] += delta2;
            sumEprod[r]  += deltaEprod;
            sumD2degC[r] += deltaD2degC;
        }

        long sumDegIn2(int r) { return sumDegIn2[find(r)]; }
        double sumEprod(int r) { return sumEprod[find(r)]; }
        double sumD2degC(int r) { return sumD2degC[find(r)]; }
        int[] nodesOf(int r) { return members[find(r)].toArray(); }
    }

    // Simple dynamic int list (no boxing)
    static final class IntList {
        int[] a; int sz;
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
        public final int[] nodes;    // 0-based node ids
        public final int nC;         // |C|
        public final long sumDegIn;  // 2 * |E(C)|
        public final double Q;       // EXACT: d^T L_C d

        public SnapshotDTO(int[] nodes, int nC, long sumDegIn, double Q) {
            this.nodes = nodes;
            this.nC = nC;
            this.sumDegIn = sumDegIn;
            this.Q = Q;
        }
    }
}
