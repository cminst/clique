package UsacoProbs;

import java.io.*;
import java.util.*;
import java.util.Locale;

public class clique2_mk_benchmark_accuracy {
    static int n, m;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java clique2_mk_benchmark_accuracy <epsilon> <inputfile>");
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
        Result res = runLaplacianRMC(adj, n, EPS);
        long t1 = System.nanoTime();

        System.out.printf(Locale.US, "Runtime: %.3f ms%n", (t1 - t0) / 1_000_000.0);
    }

    /**
     * Optimized O(Mk) algorithm for L-RMC.
     * @param adj Adjacency list of the graph (nodes are 1-indexed)
     * @param numNodes The total number of nodes in the graph (n)
     * @param EPS A small constant to prevent division by zero
     * @return A Result object containing the best score and the set of nodes in the best component.
     */
    static Result runLaplacianRMC(List<Integer>[] adj, int numNodes, double EPS) {
        // -------- Phase 1: peeling (This part was correct) --------
        int[] currentDegrees = new int[numNodes + 1];
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        for (int i = 1; i <= numNodes; i++) {
            if (adj[i] != null) {
                currentDegrees[i] = adj[i].size();
                pq.add(new Pair(i, currentDegrees[i]));
            }
        }

        Deque<Integer> peelOrder = new ArrayDeque<>(numNodes);
        boolean[] removed = new boolean[numNodes + 1];

        while (!pq.isEmpty()) {
            Pair p = pq.poll();
            if (removed[p.node]) continue;

            removed[p.node] = true;
            peelOrder.push(p.node); // This is the peeling order

            if (adj[p.node] == null) continue;
            for (int v : adj[p.node]) {
                if (!removed[v]) {
                    currentDegrees[v]--;
                    pq.add(new Pair(v, currentDegrees[v]));
                }
            }
        }

        // Reverse peeling order to get addOrder
        int[] addOrder = new int[peelOrder.size()];
        int[] idx = new int[numNodes + 1];
        for (int t = 0; t < addOrder.length; t++) {
            int u = peelOrder.pop();
            addOrder[t] = u;
            idx[u] = t;
        }

        // -------- Phase 1.5: orient edges (This part was correct) --------
        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] succ = new ArrayList[numNodes + 1];
        @SuppressWarnings("unchecked")
        ArrayList<Integer>[] pred = new ArrayList[numNodes + 1];
        for (int i = 1; i <= numNodes; i++) { succ[i] = new ArrayList<>(); pred[i] = new ArrayList<>(); }

        for (int u = 1; u <= numNodes; u++) {
            if (adj[u] == null) continue;
            for (int v : adj[u]) {
                if (u < v) { // Process each edge once
                    if (idx[u] < idx[v]) {
                        succ[u].add(v); pred[v].add(u);
                    } else {
                        succ[v].add(u); pred[u].add(v);
                    }
                }
            }
        }
        for (int v = 1; v <= numNodes; v++) {
            succ[v].sort(Comparator.comparingInt(w -> idx[w]));
        }

        // -------- Phase 2: reverse reconstruction (CORRECTED LOGIC) --------
        DSU dsu = new DSU(numNodes);
        int[] deg = new int[numNodes + 1];
        long[] predSum = new long[numNodes + 1];

        double bestScore = -1.0;
        Set<Integer> bestComponent = new HashSet<>();

        final SumSucc sumSucc = new SumSucc(succ, idx, deg);

        for (int u : addOrder) {
            dsu.makeIfNeeded(u);
            long Su = 0L;
            final int Tu = idx[u];

            // 1. Add node u and all its back-edges, updating Q incrementally
            for (int v : pred[u]) {
                long a = deg[u], b = deg[v];
                long Sv = predSum[v] + sumSucc.until(v, Tu);

                // Calculate the change in Q from adding edge (u, v) using Equation (3)
                double deltaQ = (a - b) * (a - b)
                              + (2.0 * a * a - 2.0 * Su + a)
                              + (2.0 * b * b - 2.0 * Sv + b);

                int ru = dsu.find(u);
                int rv = dsu.find(v);
                int r = dsu.union(ru, rv);

                // Add the change to the Q of the merged component
                dsu.Q[r] += deltaQ;

                // Update degrees and helper sums for the *next* edge calculation
                deg[u]++; deg[v]++;
                for (int y : succ[u]) predSum[y]++;
                for (int y : succ[v]) predSum[y]++;
                Su += deg[v]; // Su tracks sum of degrees of u's already-added neighbors
            }

            // 2. Now that u is fully integrated, score its component
            int r_final = dsu.find(u);
            List<Integer> compNodes = dsu.componentNodes.get(r_final);
            int nc = compNodes.size();

            if (nc > 1) { // Only score non-trivial components
                // This is the correct surrogate score from Equation (2)
                double score = nc / (dsu.Q[r_final] + EPS);

                if (score > bestScore) {
                    bestScore = score;
                    bestComponent = new HashSet<>(compNodes);
                }
            }
        }

        Result out = new Result();
        out.bestScore = bestScore;
        out.bestComponent = bestComponent;
        return out;
    }

    // ---------- Helpers (can be nested or in the same file) ----------
    static class Result {
        double bestScore;
        Set<Integer> bestComponent;
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
        final double[] Q; // Laplacian energy d^T L d
        final Map<Integer, List<Integer>> componentNodes;

        DSU(int n) {
            parent = new int[n + 1];
            Q = new double[n + 1];
            componentNodes = new HashMap<>();
        }

        void makeIfNeeded(int v) {
            if (parent[v] == 0) { // More robust check
                parent[v] = v;
                Q[v] = 0.0;
                List<Integer> nodes = new ArrayList<>();
                nodes.add(v);
                componentNodes.put(v, nodes);
            }
        }

        int find(int v) {
            if (parent[v] != v) parent[v] = find(parent[v]);
            return parent[v];
        }

        int union(int a, int b) {
            a = find(a);
            b = find(b);
            if (a == b) return a;
            if (componentNodes.get(a).size() < componentNodes.get(b).size()) { int t = a; a = b; b = t; }

            parent[b] = a;
            componentNodes.get(a).addAll(componentNodes.get(b));
            componentNodes.remove(b);
            Q[a] += Q[b];
            return a;
        }
    }

    static final class SumSucc {
        final ArrayList<Integer>[] succ;
        final int[] idx;
        final int[] deg;

        SumSucc(ArrayList<Integer>[] succ, int[] idx, int[] deg) {
            this.succ = succ; this.idx = idx; this.deg = deg;
        }

        long until(int v, int T) {
            long s = 0L;
            for (int w : succ[v]) {
                if (idx[w] >= T) break;
                s += deg[w];
            }
            return s;
        }
    }
}
