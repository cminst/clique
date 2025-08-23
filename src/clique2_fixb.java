import java.io.*;
import java.util.*;

/**
 * RMC with Laplacian surrogate (Fix B: incremental reverse updates).
 *
 * Input: args[0] = epsilon (double), args[1] = path to input file
 * File format:
 *   n m
 *   u1 v1
 *   ...
 *   um vm
 */
public class clique2_fixb {
    static int n, m;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java clique2_fixb <epsilon> <inputfile>");
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

        List<Integer>[] adj = new ArrayList[n + 1];
        for (int i = 1; i <= n; i++) adj[i] = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            int a = r.nextInt(), b = r.nextInt();
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
        // Phase 1: peel by nondecreasing degree with stale-check heap
        int[] deg = new int[n + 1];
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        for (int i = 1; i <= n; i++) {
            deg[i] = adj[i].size();
            pq.add(new Pair(i, deg[i]));
        }
        Deque<Pair> stack = new ArrayDeque<>(n);
        while (!pq.isEmpty()) {
            Pair p = pq.poll();
            if (p.degree != deg[p.node]) continue; // stale
            stack.push(p);
            for (int v : adj[p.node]) {
                if (deg[v] > 0) {
                    deg[v]--;
                    pq.add(new Pair(v, deg[v]));
                }
            }
            deg[p.node] = 0;
        }

        // Phase 2: reverse reconstruction with incremental Laplacian bookkeeping
        DSU dsu = new DSU(n);
        boolean[] inGraph = new boolean[n + 1];

        // Node-local state inside the evolving graph
        int[] d = new int[n + 1];                 // internal degree
        long[] sumNbrDeg = new long[n + 1];       // sum of internal-neighbor degrees

        // Component-local energy E[root]
        long[] compEnergy = new long[n + 1];

        // For fast membership checks within components
        int[] rootSeenStamp = new int[n + 1];
        int stamp = 1;

        double bestSL = 0.0;
        int bestRoot = 0;

        while (!stack.isEmpty()) {
            Pair item = stack.pop();
            int u = item.node;

            // Gather already-in neighbors
            List<Integer> nbrs = new ArrayList<>();
            for (int v : adj[u]) {
                if (inGraph[v]) nbrs.add(v);
            }

            // Union u into the component(s) of its processed neighbors
            int root = u;
            dsu.makeIfNeeded(u);
            for (int v : nbrs) {
                root = dsu.union(root, v);
            }
            root = dsu.find(root);

            // Merge component energies from distinct neighbor roots (sum once per root)
            long mergedEnergy = compEnergy[root];
            if (!nbrs.isEmpty()) {
                long add = 0;
                for (int v : nbrs) {
                    int rv = dsu.find(v);
                    if (rootSeenStamp[rv] != stamp) {
                        rootSeenStamp[rv] = stamp;
                        add += compEnergy[rv];
                    }
                }
                stamp++;
                mergedEnergy = add; // all old parts will now be represented by root
            }

            // Insert u and add edges (u, v) incrementally
            // For each new edge, update energy using endpoint-only data, then fix the
            // node-local sums for future steps (touch only current internal neighbors).
            inGraph[u] = true;

            // Ensure root's energy starts at mergedEnergy
            compEnergy[root] = mergedEnergy;

            // Process each neighbor; maintain current root because unions may attach via u
            for (int v : nbrs) {
                // connect u-v inside the same component
                int rBefore = dsu.find(u);
                int rOther = dsu.find(v);
                if (rBefore != rOther) {
                    // Should not happen because we unioned above, but defend anyway
                    int rNew = dsu.union(u, v);
                    root = rNew;
                } else {
                    root = rBefore;
                }

                // 1) Add new edge term
                long du = d[u], dv = d[v];
                long delta = (du - dv) * (du - dv);

                // 2) Bump energy for existing edges incident to u
                // Δ_u = Σ_{x in N_in(u)} [2(du - d[x]) + 1]
                // which equals 2*du*deg_u - 2*sumNbrDeg[u] + deg_u
                long deg_u = du;
                delta += 2 * du * deg_u - 2 * sumNbrDeg[u] + deg_u;

                // 3) Symmetric bump for v
                long deg_v = dv;
                delta += 2 * dv * deg_v - 2 * sumNbrDeg[v] + deg_v;

                compEnergy[root] += delta;

                // 4) Update neighbor sums for endpoints, then increment degrees
                // First, add each other’s current degree to the endpoint’s neighbor-degree sum
                sumNbrDeg[u] += dv;
                sumNbrDeg[v] += du;

                // Now the degrees actually increase by 1
                d[u] = (int) (du + 1);
                d[v] = (int) (dv + 1);

                // 5) Because d[u] and d[v] increased by 1, every current internal neighbor
                // of u and v should see its sumNbrDeg[*] go up by 1 once.
                // We touch only internal neighbors, not the whole adjacency.
                for (int x : adj[u]) if (inGraph[x] && dsu.find(x) == root) sumNbrDeg[x] += 1;
                for (int y : adj[v]) if (inGraph[y] && dsu.find(y) == root) sumNbrDeg[y] += 1;
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

        Result out = new Result();
        out.bestSL = bestSL;
        out.bestRoot = bestRoot;
        return out;
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
