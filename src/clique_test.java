import java.util.*;

public class clique_test {
    static int n, m;

    public static void main(String[] args) throws Exception {
        long t0 = System.nanoTime();
        Result res = new Result();
        long t1 = System.nanoTime();

        System.out.printf(Locale.US, "%.6f, %d%n", res.bestSL, res.bestRoot);
        System.out.printf(Locale.US, "Runtime: %.3f ms%n", (t1 - t0) / 1_000_000.0);
    }

    static Result runLaplacianRMC(List<Integer>[] adj, double EPS) {
        return new Result();
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
