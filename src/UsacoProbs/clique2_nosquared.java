package UsacoProbs;

import java.io.*;
import java.util.*;

public class clique2_nosquared {
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

        @SuppressWarnings("unchecked")
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

        // --- Phase 2: reverse reconstruction with batched Laplacian update ---

        DSU dsu = new DSU(n);
        boolean[] inGraph = new boolean[n + 1];

        // Internal degree inside the evolving graph
        int[] d = new int[n + 1];

        // Component energy E[root]
        long[] compEnergy = new long[n + 1];

        // Stamps for distinct-root accumulation and for marking A
        int[] rootSeenStamp = new int[n + 1];
        int stamp = 1;

        int[] inAStamp = new int[n + 1];
        int aStamp = 1;

        double bestSL = 0.0;
        int bestRoot = 0;

        while (!stack.isEmpty()) {
            Pair item = stack.pop();
            int u = item.node;

            // Collect already-in neighbors A
            List<Integer> A = new ArrayList<>();
            for (int v : adj[u]) if (inGraph[v]) A.add(v);

            // Sum energies from distinct neighbor roots BEFORE union
            long mergedEnergy = 0L;
            stamp++;
            for (int v : A) {
                int rv = dsu.find(v);
                if (rootSeenStamp[rv] != stamp) {
                    rootSeenStamp[rv] = stamp;
                    mergedEnergy += compEnergy[rv];
                }
            }

            // Mark A for O(1) membership tests
            aStamp++;
            for (int v : A) inAStamp[v] = aStamp;

            // Compute delta on old edges touching A, but only within current in-graph
            long deltaOld = 0L;
            for (int w : A) {
                for (int x : adj[w]) {
                    if (!inGraph[x]) continue;           // not in current graph
                    if (inAStamp[x] == aStamp) continue; // x also in A, change = 0
                    deltaOld += 2L * ((long) d[w] - (long) d[x]) + 1L;
                }
            }

            // Contribution from the |A| new edges (u, w) with w in A
            int degU = A.size();
            long deltaNew = 0L;
            for (int w : A) {
                long t = (long) degU - ((long) d[w] + 1L);
                deltaNew += t * t;
            }

            // Create u and union with all neighbors in A
            dsu.makeIfNeeded(u);
            int root = u;
            for (int v : A) root = dsu.union(root, v);

            // Activate u and update degrees
            inGraph[u] = true;
            d[u] = degU;
            for (int w : A) d[w]++;

            // The new component's energy
            compEnergy[root] = mergedEnergy + deltaOld + deltaNew;

            // Score
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
