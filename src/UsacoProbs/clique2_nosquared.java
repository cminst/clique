package UsacoProbs;

import java.io.*;
import java.util.*;

public class clique2_nosquared {
    static int n, m;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java clique2_nosquared <epsilon> <inputfile>");
        }
        final double EPS = Double.parseDouble(args[0]);

        Scanner r;
        try {
            r = new Scanner(new FileReader(args[1]));
        } catch (IOException e) {
            System.err.println("Could not open " + args[1] + ". Falling back to stdin.");
            r = new Scanner(System.in);
        }

        n = r.nextInt();
        m = r.nextInt();

        @SuppressWarnings("unchecked")
        List<Integer>[] adj = new ArrayList[n + 1];
        for (int i = 1; i <= n; i++) adj[i] = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            int a = r.nextInt(), b = r.nextInt();
            if (a == b) continue; // ignore self-loops if present
            adj[a].add(b);
            adj[b].add(a);
        }
        r.close();

        long t0 = System.nanoTime();
        Result res = runLaplacianRMC(adj, EPS);
        long t1 = System.nanoTime();

        System.out.printf(Locale.US, "%.6f, %d%n", res.bestSL, res.bestRoot);
        System.out.printf(Locale.US, "Runtime: %.3f ms%n", (t1 - t0) / 1_000_000.0);
    }

    static Result runLaplacianRMC(List<Integer>[] adj, double EPS) {
        // -------------------
        // Phase 1: peel by nondecreasing degree with stale-check heap
        // Also record removal rank to create a κ-orientation later.
        // -------------------
        int[] degWork = new int[n + 1];
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        for (int i = 1; i <= n; i++) {
            degWork[i] = adj[i].size();
            pq.add(new Pair(i, degWork[i]));
        }

        Deque<Integer> stack = new ArrayDeque<>(n);
        int[] rank = new int[n + 1]; // removal order index (smaller => removed earlier)
        int remIdx = 0;

        while (!pq.isEmpty()) {
            Pair p = pq.poll();
            if (p.degree != degWork[p.node]) continue; // stale
            int u = p.node;
            stack.push(u);
            rank[u] = remIdx++;

            for (int v : adj[u]) {
                if (degWork[v] > 0) {
                    degWork[v]--;
                    pq.add(new Pair(v, degWork[v]));
                }
            }
            degWork[u] = 0;
        }

        // -------------------
        // Build κ-orientation from the peel order:
        // direct every {u,v} from the earlier-removed endpoint to the later-removed.
        // Out-degree is bounded by graph degeneracy κ.
        // -------------------
        @SuppressWarnings("unchecked")
        List<Integer>[] out = new ArrayList[n + 1];
        for (int i = 1; i <= n; i++) out[i] = new ArrayList<>();

        for (int u = 1; u <= n; u++) {
            for (int v : adj[u]) if (u < v) { // process each undirected edge once
                if (rank[u] < rank[v]) out[u].add(v);
                else out[v].add(u);
            }
        }

        // -------------------
        // Phase 2: reverse reconstruction with exact Laplacian energy,
        // using orientation to keep updates O(m κ).
        // -------------------
        DSU dsu = new DSU(n);
        boolean[] inGraph = new boolean[n + 1];

        int[] d = new int[n + 1];          // internal degree in the evolving component
        long[] sumIn = new long[n + 1];    // Σ d[p] for p that are incoming to node (p -> u) and in same component
        long[] sumOutProc = new long[n + 1]; // Σ d[x] over out-neighbors x that are ALREADY connected to u

        long[] compEnergy = new long[n + 1]; // E[root] for each DSU root

        int[] rootSeenStamp = new int[n + 1];
        int stamp = 1;

        double bestSL = 0.0;
        int bestRoot = 0;

        while (!stack.isEmpty()) {
            int u = stack.pop();
            inGraph[u] = true;

            // Neighbors already in the graph are exactly out[u]
            List<Integer> nbrs = out[u];

            // Sum energies from distinct neighbor components BEFORE union
            long mergedEnergy = 0L;
            if (!nbrs.isEmpty()) {
                int s = stamp++;
                for (int v : nbrs) {
                    if (!inGraph[v]) continue; // defensive; should always be true
                    int rv = dsu.find(v);
                    if (rootSeenStamp[rv] != s) {
                        rootSeenStamp[rv] = s;
                        mergedEnergy += compEnergy[rv];
                    }
                }
            }

            // Union u with all processed neighbors
            dsu.makeIfNeeded(u);
            int root = u;
            for (int v : nbrs) {
                if (!inGraph[v]) continue;
                root = dsu.union(root, v);
            }
            root = dsu.find(root);
            compEnergy[root] = mergedEnergy;

            // Insert edges (u, v) one by one, updating energy exactly.
            for (int v : nbrs) {
                if (!inGraph[v]) continue; // defensive
                // Degrees before the new edge
                long du = d[u];
                long dv = d[v];

                // Current neighbor-degree sums (only already-connected neighbors count)
                long sumNbr_u = sumIn[u] + sumOutProc[u];
                long sumNbr_v = sumIn[v] + sumOutProc[v];

                // 1) New edge term
                long delta = (du - dv) * (du - dv);

                // 2) Bump for edges incident to u (excluding v, since not connected yet)
                //    Σ_{x in N(u)} [2(du - d[x]) + 1]  where N(u) = already-connected internal neighbors
                long deg_u = du;
                delta += 2 * du * deg_u - 2 * sumNbr_u + deg_u;

                // 3) Symmetric bump for v
                long deg_v = dv;
                delta += 2 * dv * deg_v - 2 * sumNbr_v + deg_v;

                compEnergy[root] += delta;

                // 4) Update per-node neighbor-degree trackers for the new adjacency
                //    v becomes an out-neighbor that is "processed" for u,
                //    and u becomes an in-neighbor for v.
                sumOutProc[u] += dv;
                sumIn[v] += du;

                // 5) Now the degrees increase by 1
                d[u] = (int) (du + 1);
                d[v] = (int) (dv + 1);

                // 6) Propagate the +1 only along out-edges, inside the current component
                //    This accounts for every adjacent edge's degree change, exactly.
                for (int w : out[u]) {
                    if (inGraph[w] && dsu.find(w) == root) sumIn[w] += 1;
                }
                for (int w : out[v]) {
                    if (inGraph[w] && dsu.find(w) == root) sumIn[w] += 1;
                }
            }

            // Score the component containing u
            int compRoot = dsu.find(u);
            int compSize = dsu.size[compRoot];
            double sL = compSize / (compEnergy[compRoot] + EPS);
            if (sL > bestSL) {
                bestSL = sL;
                bestRoot = compRoot;
            }
        }

        Result outRes = new Result();
        outRes.bestSL = bestSL;
        outRes.bestRoot = bestRoot;
        return outRes;
    }

    // ---------- Helpers ----------

    static class Result {
        double bestSL;
        int bestRoot;
    }

    static class Pair implements Comparable<Pair> {
        final int node, degree;
        Pair(int node, int degree) { this.node = node; this.degree = degree; }
        public int compareTo(Pair o) {
            if (degree != o.degree) return Integer.compare(degree, o.degree);
            return Integer.compare(node, o.node);
        }
    }

    static class DSU {
        final int[] parent;
        final int[] size;
        final boolean[] made;

        DSU(int n) {
            parent = new int[n + 1];
            size = new int[n + 1];
            made = new boolean[n + 1];
        }
        void makeIfNeeded(int v) {
            if (!made[v]) {
                made[v] = true;
                parent[v] = v;
                size[v] = 1;
            }
        }
        int find(int v) {
            if (!made[v]) return v; // treat as isolated until made
            if (parent[v] != v) parent[v] = find(parent[v]);
            return parent[v];
        }
        int union(int a, int b) {
            makeIfNeeded(a);
            makeIfNeeded(b);
            int ra = find(a), rb = find(b);
            if (ra == rb) return ra;
            if (size[ra] < size[rb]) { int t = ra; ra = rb; rb = t; }
            parent[rb] = ra;
            size[ra] += size[rb];
            return ra;
        }
    }
}
