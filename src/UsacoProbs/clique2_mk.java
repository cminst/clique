package UsacoProbs;

import java.io.*;
import java.util.*;
import java.util.Locale;

public class clique2_mk {
    static int n, m;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java clique2_mk <epsilon> <inputfile>");
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
        Result res = runLaplacianRMC(adj, EPS);  // <- optimized O(Mk)
        long t1 = System.nanoTime();

        System.out.printf(Locale.US, "%.6f, %d%n", res.bestSL, res.bestRoot);
        System.out.printf(Locale.US, "Runtime: %.3f ms%n", (t1 - t0) / 1_000_000.0);
    }

    /** Optimized O(Mk) algorithm using reverse-peeling orientation + pred_sum pushes. */
    static Result runLaplacianRMC(List<Integer>[] adj, double EPS) {
        // Phase 1: peeling (same as before)
        int[] deg0 = new int[n + 1];
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        for (int i = 1; i <= n; i++) {
            deg0[i] = adj[i].size();
            pq.add(new Pair(i, deg0[i]));
        }
        Deque<Integer> peelStack = new ArrayDeque<>(n); // store nodes only
        while (!pq.isEmpty()) {
            Pair p = pq.poll();
            if (p.degree != deg0[p.node]) continue; // stale
            peelStack.push(p.node);
            for (int v : adj[p.node]) {
                if (deg0[v] > 0) {
                    deg0[v]--;
                    pq.add(new Pair(v, deg0[v]));
                }
            }
            deg0[p.node] = 0;
        }

        // Build addition order and index
        int[] addOrder = new int[n];
        int[] idx = new int[n + 1];
        for (int t = 0; t < n; t++) {
            int u = peelStack.pop(); // reverse-peeling (addition order)
            addOrder[t] = u;
            idx[u] = t;
        }

        // Phase 1.5: orient edges by idx and sort successors
        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] succ = new ArrayList[n + 1];
        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] pred = new ArrayList[n + 1];
        for (int i = 1; i <= n; i++) { succ[i] = new ArrayList<>(); pred[i] = new ArrayList<>(); }

        for (int u = 1; u <= n; u++) {
            for (int v : adj[u]) {
                if (u < v) { // handle undirected edge once
                    if (idx[u] < idx[v]) {
                        succ[u].add(v);
                        pred[v].add(u);
                    } else {
                        succ[v].add(u);
                        pred[u].add(v);
                    }
                }
            }
        }
        for (int v = 1; v <= n; v++) {
            if (succ[v].size() > 1) {
                succ[v].sort(Comparator.comparingInt(w -> idx[w]));
            }
            // pred[v] need not be sorted
        }

        // Phase 2: reverse reconstruction with O(k) per edge
        DSU dsu = new DSU(n); // tracks parent, size, and Q (double)
        int[] deg = new int[n + 1];          // current degree
        long[] predSum = new long[n + 1];    // sum of degrees of predecessors

        double bestSL = 0.0;
        int bestRoot = 0;

        // helper: sum of degrees of active successors of v whose idx < T
        final SumSucc sumSucc = new SumSucc(succ, idx, deg);

        for (int u : addOrder) {
            dsu.makeIfNeeded(u); // create singleton component
            // Single-node score (Q=0)
            {
                int ru = dsu.find(u);
                double sL = dsu.size[ru] / (dsu.Q[ru] + EPS);
                if (sL > bestSL) { bestSL = sL; bestRoot = ru; }
            }

            long Su = 0L; // running sum over degrees of neighbors already attached to u
            final int Tu = idx[u];

            // connect u to all its predecessors (earlier neighbors)
            for (int v : pred[u]) {
                long a = deg[u];
                long b = deg[v];

                // S_v = pred_sum[v] + sum of deg[w] for successors w of v with idx[w] < idx[u]
                long Sv = predSum[v] + sumSucc.until(v, Tu);

                long dQu = 2L * a * a - 2L * Su + a;
                long dQv = 2L * b * b - 2L * Sv + b;
                long edgeTerm = (a - b) * (a - b);

                int ru = dsu.find(u);
                int rv = dsu.find(v);

                dsu.Q[ru] += (double) dQu;
                dsu.Q[rv] += (double) dQv;

                int r;
                if (ru != rv) {
                    r = dsu.union(ru, rv);
                    dsu.Q[r] += (double) edgeTerm;
                } else {
                    r = ru;
                    dsu.Q[r] += (double) edgeTerm;
                }

                // score after this edge activation
                double sL = dsu.size[r] / (dsu.Q[r] + EPS);
                if (sL > bestSL) { bestSL = sL; bestRoot = r; }

                // degree increments
                deg[u] += 1;
                deg[v] += 1;

                // push +1 to predSum of successors (outdegree ≤ k)
                for (int y : succ[u]) predSum[y] += 1;
                for (int y : succ[v]) predSum[y] += 1;

                // maintain Su: add deg[v] AFTER its increment
                Su += deg[v];
            }
        }

        Result out = new Result();
        out.bestSL = bestSL;
        out.bestRoot = bestRoot;
        return out;
    }

    // Small helper for successor-degree partial sums
    static final class SumSucc {
        final ArrayList<Integer>[] succ;
        final int[] idx;
        final int[] deg;

        SumSucc(ArrayList<Integer>[] succ, int[] idx, int[] deg) {
            this.succ = succ; this.idx = idx; this.deg = deg;
        }

        /** Sum of deg[w] over successors w of v with idx[w] < T (succ[v] sorted by idx). */
        long until(int v, int T) {
            long s = 0L;
            final ArrayList<Integer> sv = succ[v];
            final int sz = sv.size();
            for (int i = 0; i < sz; i++) {
                int w = sv.get(i);
                if (idx[w] >= T) break;
                s += deg[w];
            }
            return s;
        }
    }

    // Helpers

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

    /** DSU that also tracks component Laplacian Q as double. */
    static class DSU {
        final int[] parent;
        final int[] size;
        final boolean[] made;
        final double[] Q;

        DSU(int n) {
            parent = new int[n + 1];
            size = new int[n + 1];
            made = new boolean[n + 1];
            Q = new double[n + 1];
        }
        void makeIfNeeded(int v) {
            if (!made[v]) {
                made[v] = true;
                parent[v] = v;
                size[v] = 1;
                Q[v] = 0.0;
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
            Q[ra] += Q[rb];
            return ra;
        }
    }
}
