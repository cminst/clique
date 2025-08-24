// ===============================================
// >>> DO NOT USE THIS FOR SPEED BENCHMARKING! <<<
// ===============================================
package UsacoProbs;

import java.io.*;
import java.util.*;

public class clique2_mk_benchmark_accuracy {
    static int n, m;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java clique2_mk_benchmark_accuracy <epsilon> <inputfile> [outputfile]");
            return;
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

        // Add debug output
        int clusterSize = 0;
        for (int i = 1; i <= n; i++) {
            if (res.bestMask[i]) clusterSize++;
        }
        System.err.println("L-RMC: Best SL = " + res.bestSL + ", cluster size = " + clusterSize + " / " + n);

        if (args.length >= 3) {
            try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(args[2])))) {
                // The harness expects 0-based node IDs from 0 to n-1.
                // Internal nodes are 1 to n. We must convert.
                for (int i = 1; i <= n; i++) { // Iterate over valid nodes 1..n
                    // Write 0-based ID (i-1) and its membership status from bestMask[i]
                    pw.println((i - 1) + " " + (res.bestMask[i] ? 1 : 0));
                }
            }
        }

        System.out.printf(Locale.US, "Best SL: %.6f, in component of node %d%n", res.bestSL, res.bestRoot);
        System.out.printf(Locale.US, "Runtime: %.3f ms%n", (t1 - t0) / 1_000_000.0);
    }

    static Result runLaplacianRMC(List<Integer>[] adj, double EPS) {
        // -------- Phase 1: peeling --------
        int[] deg0 = new int[n + 1];
        boolean[] bestMask = new boolean[n + 1];
        PriorityQueue<Pair> pq = new PriorityQueue<>();

        for (int i = 1; i <= n; i++) {
            deg0[i] = adj[i].size();
            pq.add(new Pair(i, deg0[i]));
        }
        Deque<Integer> peelStack = new ArrayDeque<>(n);
        while (!pq.isEmpty()) {
            Pair p = pq.poll();
            if (p.degree != deg0[p.node]) continue;
            peelStack.push(p.node);
            for (int v : adj[p.node]) {
                if (deg0[v] > 0) {
                    deg0[v]--;
                    pq.add(new Pair(v, deg0[v]));
                }
            }
            deg0[p.node] = -1; // Mark as fully peeled
        }

        int[] addOrder = new int[n];
        int[] idx = new int[n + 1];
        for (int t = 0; t < n; t++) {
            int u = peelStack.pop();
            addOrder[t] = u;
            idx[u] = t;
        }

        // -------- Phase 1.5: orient edges --------
        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] succ = new ArrayList[n + 1];
        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] pred = new ArrayList[n + 1];
        for (int i = 1; i <= n; i++) { succ[i] = new ArrayList<>(); pred[i] = new ArrayList<>(); }

        for (int u = 1; u <= n; u++) {
            for (int v : adj[u]) {
                if (idx[u] < idx[v]) {
                    succ[u].add(v);
                    pred[v].add(u);
                }
            }
        }
        for (int v = 1; v <= n; v++) {
            if (succ[v].size() > 1) {
                succ[v].sort(Comparator.comparingInt(w -> idx[w]));
            }
        }

        // -------- Phase 2: reverse reconstruction --------
        DSU dsu = new DSU(n);
        int[] deg = new int[n + 1];
        long[] predSum = new long[n + 1];

        double bestSL = 0.0;
        int bestRoot = -1;

        final SumSucc sumSucc = new SumSucc(succ, idx, deg);

        for (int u : addOrder) {
            dsu.makeIfNeeded(u);
            long Su = 0L;
            final int Tu = idx[u];

            for (int v : pred[u]) {
                long a = deg[u];
                long b = deg[v];
                long Sv = predSum[v] + sumSucc.until(v, Tu);

                long dQu = 2L * a * a - 2L * Su + a;
                long dQv = 2L * b * b - 2L * Sv + b;
                long edgeTerm = (a - b) * (a - b);

                int ru = dsu.find(u);
                int rv = dsu.find(v);

                // Temporarily apply changes to copies to score before committing
                double tempQ_ru = dsu.Q[ru] + dQu;
                double tempQ_rv = dsu.Q[rv] + dQv;

                int r_new;
                double Q_new;
                int size_new;

                if (ru != rv) {
                    if (dsu.size[ru] < dsu.size[rv]) { int t = ru; ru = rv; rv = t; } // Keep consistent parent
                    r_new = ru;
                    size_new = dsu.size[ru] + dsu.size[rv];
                    Q_new = tempQ_ru + tempQ_rv + edgeTerm;
                } else {
                    r_new = ru;
                    size_new = dsu.size[ru];
                    // dQu and dQv are double-counted on the same component, but edgeTerm is new
                    Q_new = dsu.Q[ru] + dQu + dQv + edgeTerm;
                }

                double sL = size_new / (Q_new + EPS);
                if (sL > bestSL) {
                    bestSL = sL;
                    bestRoot = r_new;

                    // Snapshot membership.
                    Arrays.fill(bestMask, false);
                    for (int i = 1; i <= n; i++) { // Iterate over valid nodes 1..n
                        // Check if node i belongs to the component(s) that are about to be merged
                        if (dsu.find(i) == ru || dsu.find(i) == rv) {
                            bestMask[i] = true;
                        }
                    }
                }

                // Commit changes to DSU
                dsu.union(u, v);
                int r_final = dsu.find(u);
                dsu.Q[r_final] = Q_new; // Update Q after merge

                deg[u]++;
                deg[v]++;

                for (int y : succ[u]) predSum[y]++;
                for (int y : succ[v]) predSum[y]++;

                Su += deg[v];
            }
        }

        Result out = new Result();
        out.bestSL = bestSL;
        out.bestRoot = bestRoot;
        out.bestMask = bestMask;
        return out;
    }

    // ---------- Small helper for successor-degree partial sums ----------
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

    // ---------- Helpers ----------

    static class Result {
        double bestSL;
        int bestRoot;
        boolean[] bestMask; //  length n, true if node is in best cluster
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
