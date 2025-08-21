package UsacoProbs;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class clique2_lct {
    public static int n;

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader(args[1]));
        double epsilon = Double.parseDouble(args[0]);

        n = r.nextInt();
        int m = r.nextInt();

        long start = System.nanoTime();

        // --- Adjacency List ---
        Map<Integer, List<Integer>> adj = new HashMap<>();
        for (int i = 1; i <= n; i++) adj.put(i, new ArrayList<>());
        for (int i = 0; i < m; i++) {
            int u = r.nextInt();
            int v = r.nextInt();
            adj.get(u).add(v);
            adj.get(v).add(u);
        }

        // --- Phase 1: Peeling (Unchanged, but using a more robust PQ method) ---
        int[] currentDegrees = new int[n + 1];
        boolean[] removed = new boolean[n + 1];
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        Stack<Integer> removalOrder = new Stack<>();

        for(int i = 1; i <= n; i++) {
            currentDegrees[i] = adj.get(i).size();
            pq.add(new Pair(i, currentDegrees[i]));
        }

        int removedCount = 0;
        while(removedCount < n) {
            Pair p = pq.poll();
            int u = p.node;

            if (removed[u]) continue; // Skip stale entries in the PQ

            removed[u] = true;
            removalOrder.push(u);
            removedCount++;

            for (int v : adj.get(u)) {
                if (!removed[v]) {
                    currentDegrees[v]--;
                    pq.add(new Pair(v, currentDegrees[v]));
                }
            }
        }

        // --- Phase 2: Rebuilding with Link-Cut Tree ---
        double sMax = 0;
        int uStar = 0;

        LinkCutTree lct = new LinkCutTree(n);
        // DSU is still useful for tracking component-wide stats simply
        DSU dsu = new DSU(n);
        Map<Integer, Set<Integer>> nonTreeEdges = new HashMap<>();
        for (int i = 1; i <= n; i++) nonTreeEdges.put(i, new HashSet<>());

        boolean[] readded = new boolean[n + 1];

        while (!removalOrder.isEmpty()) {
            int u = removalOrder.pop();
            readded[u] = true;

            // 1. Initialize u in the data structures
            lct.makeTree(u);

            Set<Integer> readdedNeighbors = new HashSet<>();
            for (int v : adj.get(u)) {
                if (readded[v]) {
                    readdedNeighbors.add(v);
                }
            }

            // 2. Initial properties for the new component centered at u
            int componentRoot = dsu.find(u);
            long mergedLaplacian = dsu.laplacianSum[componentRoot];
            long mergedDegreeSum = dsu.internalDegreeSum[componentRoot];

            // Set initial degree of u (it will be updated)
            lct.updatePoint(u, readdedNeighbors.size());
            mergedDegreeSum += readdedNeighbors.size();

            // 3. Process each new edge (u, v)
            for (int v : readdedNeighbors) {
                // --- A: Update for degree[v] increasing by 1 ---
                long old_dv = lct.getDegree(v);

                // Get sum of neighbor degrees for v BEFORE its degree changes
                long sumOfVNeighborDegrees = lct.getSumOfTreeNeighborDegrees(v);
                for (int nonTreeNeighbor : nonTreeEdges.get(v)) {
                    sumOfVNeighborDegrees += lct.getDegree(nonTreeNeighbor);
                }

                int vRoot = dsu.find(v);
                long changeInVComponentLaplacian = 2 * sumOfVNeighborDegrees - 2 * old_dv * (dsu.size[vRoot] - 1);
                dsu.laplacianSum[vRoot] += changeInVComponentLaplacian;

                // Actually update v's degree in the LCT
                lct.updatePoint(v, old_dv + 1);

                // --- B: Add contribution of the new edge (u, v) ---
                long du = lct.getDegree(u);
                long new_dv = lct.getDegree(v);
                long edgeContribution = (du - new_dv) * (du - new_dv);

                // --- C: Update topology ---
                if (lct.findRoot(u) != lct.findRoot(v)) {
                    // It's a tree edge, merge components
                    int uRoot = dsu.find(u);
                    int vRootOld = dsu.find(v);

                    // Merge DSU stats before linking
                    dsu.union(u, v);
                    int newRoot = dsu.find(u);

                    // The Laplacian sum is the sum of the parts plus the changes
                    long laplacianU = (uRoot == newRoot) ? dsu.laplacianSum[uRoot] : dsu.laplacianSum[vRootOld];
                    long laplacianV = (vRootOld == newRoot) ? dsu.laplacianSum[vRootOld] : dsu.laplacianSum[uRoot];

                    dsu.laplacianSum[newRoot] = laplacianU + laplacianV + edgeContribution;
                    dsu.internalDegreeSum[newRoot] += dsu.internalDegreeSum[vRootOld] + 1; // +1 for v's degree increase

                    lct.link(u, v);
                } else {
                    // It's a non-tree edge, update stats for the single component
                    nonTreeEdges.get(u).add(v);
                    nonTreeEdges.get(v).add(u);
                    dsu.laplacianSum[dsu.find(u)] += edgeContribution;
                    dsu.internalDegreeSum[dsu.find(u)] += 1; // for v's degree increase
                }
            }

            // 4. Calculate score for the final merged component
            int finalRoot = dsu.find(u);
            double sL = (double) dsu.size[finalRoot] / (dsu.laplacianSum[finalRoot] + epsilon);

            if (sL > sMax) {
                sMax = sL;
                uStar = finalRoot;
            }
        }

        System.out.println("Max Score (sMax): " + sMax + ", Component Root (uStar): " + uStar);
        long end = System.nanoTime();
        System.out.printf("Runtime: %.3f ms%n", (end - start) / 1_000_000.0);
    }

    // --- Helper Classes ---

    static class DSU {
        int[] parent;
        int[] size;
        long[] laplacianSum;
        long[] internalDegreeSum;

        DSU(int n) {
            parent = new int[n + 1];
            size = new int[n + 1];
            laplacianSum = new long[n + 1];
            internalDegreeSum = new long[n + 1];
            for (int i = 1; i <= n; i++) {
                parent[i] = i;
                size[i] = 1;
            }
        }
        public int find(int i) {
            if (parent[i] == i) return i;
            return parent[i] = find(parent[i]);
        }
        public void union(int i, int j) {
            int root_i = find(i);
            int root_j = find(j);
            if (root_i != root_j) {
                if (size[root_i] < size[root_j]) {
                    int temp = root_i; root_i = root_j; root_j = temp;
                }
                parent[root_j] = root_i;
                size[root_i] += size[root_j];
                // Component stats are merged manually in the main loop
            }
        }
    }

    static class Pair implements Comparable<Pair> {
        int node, degree;
        public Pair(int n, int d) { node = n; degree = d; }
        public int compareTo(Pair o) {
            if (degree != o.degree) return Integer.compare(degree, o.degree);
            return Integer.compare(node, o.node);
        }
    }
}

/**
 * An Augmented Link-Cut Tree for Dynamic Graphs.
 * This LCT maintains a spanning forest of a graph.
 * It is augmented to track the sum of node values (degrees) within subtrees,
 * which allows for efficient calculation of neighbor-degree sums.
 */
class LinkCutTree {
    Node[] nodes;

    static class Node {
        int id;
        Node p = null, l = null, r = null; // Parent, left, right in splay tree
        boolean rev = false; // Lazy flag for path reversal

        // --- Augmentation Data ---
        long value; // The node's actual degree
        long subtreeValue; // Sum of values in the splay subtree (real children)
        long virtualValue; // Sum of subtreeValues of virtual children
    }

    public LinkCutTree(int n) {
        nodes = new Node[n + 1];
        for (int i = 1; i <= n; i++) {
            nodes[i] = new Node();
            nodes[i].id = i;
        }
    }

    // --- Core Splay Tree Operations ---
    private boolean isRoot(Node x) {
        return x.p == null || (x.p.l != x && x.p.r != x);
    }

    private void push(Node x) {
        if (!x.rev) return;
        Node temp = x.l; x.l = x.r; x.r = temp;
        if (x.l != null) x.l.rev ^= true;
        if (x.r != null) x.r.rev ^= true;
        x.rev = false;
    }

    private void pull(Node x) {
        x.subtreeValue = x.value + x.virtualValue;
        if (x.l != null) x.subtreeValue += x.l.subtreeValue;
        if (x.r != null) x.subtreeValue += x.r.subtreeValue;
    }

    private void rotate(Node x) {
        Node p = x.p, gp = p.p;
        boolean isRootP = isRoot(p);

        push(p); push(x);

        if (x == p.l) {
            p.l = x.r;
            if (x.r != null) x.r.p = p;
            x.r = p;
        } else {
            p.r = x.l;
            if (x.l != null) x.l.p = p;
            x.l = p;
        }
        p.p = x;
        x.p = gp;
        if (!isRootP) {
            if (p == gp.l) gp.l = x;
            else if (p == gp.r) gp.r = x;
        }
        pull(p); pull(x);
    }

    private void splay(Node x) {
        while (!isRoot(x)) {
            Node p = x.p, gp = p.p;
            if (!isRoot(p)) push(gp);
            push(p); push(x);
            if (!isRoot(p)) {
                if ((p.l == x) == (gp.l == p)) rotate(p);
                else rotate(x);
            }
            rotate(x);
        }
        push(x);
    }

    // --- Core Link-Cut Tree Operations ---

    // Moves x to the root of its auxiliary splay tree and makes the path
    // from x to the root of the represented tree the preferred path.
    private Node access(Node x) {
        Node last = null;
        for (Node y = x; y != null; y = y.p) {
            splay(y);
            // Detach old preferred child, attach new one
            if (y.r != null) y.virtualValue += y.r.subtreeValue;
            y.r = last;
            if (last != null) y.virtualValue -= last.subtreeValue;
            pull(y);
            last = y;
        }
        splay(x);
        return last;
    }

    private void makeRoot(Node x) {
        access(x);
        x.rev ^= true;
        push(x);
    }

    public int findRoot(int u) {
        Node x = nodes[u];
        access(x);
        while (x.l != null) {
            push(x);
            x = x.l;
        }
        splay(x);
        return x.id;
    }

    public void link(int u, int v) {
        Node x = nodes[u], y = nodes[v];
        makeRoot(x);
        access(y);
        x.p = y;
        y.virtualValue += x.subtreeValue;
        pull(y);
    }

    public void updatePoint(int u, long newValue) {
        Node x = nodes[u];
        access(x);
        x.value = newValue;
        pull(x);
    }

    public long getDegree(int u) {
        return nodes[u].value;
    }

    public long getSumOfTreeNeighborDegrees(int u) {
        Node x = nodes[u];
        access(x); // This makes path to root preferred and brings x to splay root
        long sum = x.virtualValue; // Sum from non-preferred children
        if (x.l != null) sum += nodes[x.l.id].value; // Parent in represented tree
        if (x.r != null) sum += nodes[x.r.id].value; // Child on preferred path
        return sum;
    }

    public void makeTree(int u) {
        pull(nodes[u]);
    }
}