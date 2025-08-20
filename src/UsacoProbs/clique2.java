package UsacoProbs;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class clique2 {
    public static int n;
    public static double main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader(args[1]));
        double epsilon = Double.parseDouble(args[0]);

        n = r.nextInt();
        int m = r.nextInt();

        long start = System.nanoTime();

        TreeMap<Integer, ArrayList<Integer>> map = new TreeMap<>();
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        int[] degrees = new int[n+1];
        Stack<Pair> stack = new Stack<>();

        for (int i = 0; i < n; i++) {
            map.put(i + 1, new ArrayList<>());
        }

        for (int i = 0; i < m; i++) { // Read in connections
            int node1 = r.nextInt();
            int node2 = r.nextInt();
            map.get(node1).add(node2);
            map.get(node2).add(node1);
        }

        for (int i = 1; i < n + 1; i++) {
            pq.add(new Pair(i, map.get(i).size()));
            degrees[i] = map.get(i).size();
        }

        // Remove nodes one by one

        while(!pq.isEmpty()) {
            Pair minDegreeNode = pq.poll();

            // Ignore outdated nodes
            if (degrees[minDegreeNode.node] != minDegreeNode.degree) continue;

            for (int connectedNode : map.get(minDegreeNode.node)) {
                if (degrees[connectedNode] > 0) {
                    // Remove the current minDegreeNode and update neighbor's nodes

                    degrees[connectedNode] -= 1;
                    pq.add(new Pair(connectedNode, degrees[connectedNode]));
                }
            }
            degrees[minDegreeNode.node] = 0;
            stack.add(minDegreeNode);
        }

        // Add back nodes

        DSU dsu = new DSU();
        double sMax = 0;
        int uStar = 0;

        long[] currLaplacian = new long[n+1]; // Stores Laplacian values for each component
        boolean[] readded = new boolean[n+1]; // Tracks which nodes have been added back

        // Process nodes in reverse order from stack
        while (!stack.isEmpty()) {
            Pair u = stack.pop();

            // Find all neighbors that have already been processed (added back to graph)
            ArrayList<Integer> uN = new ArrayList<>();
            for (int v : map.get(u.node)) {
                if (readded[v]) {
                    uN.add(v);
                }
            }

            // Collect all unique component roots from processed neighbors
            HashSet<Integer> roots = new HashSet<>();
            for (int v : uN) {
                int rv = dsu.find(v);
                roots.add(rv);

                // Update Laplacian based on degree differences within the component
                int dv = degrees[v];
                for (int x : map.get(v)) {
                    if (readded[x]) {
                        int dx = degrees[x];
                        currLaplacian[rv] += 2L * (dv - dx) + 1L;
                    }
                }
                degrees[v] += 1;
            }

            // Update degree of current node to number of processed neighbors so far
            int du = uN.size();
            degrees[u.node] = du;

            // Calculate Laplacian contribution from new edges (based on degree differences)
            long newEdge = 0;
            for (int v : uN) {
                int dv = degrees[v];
                long degreeDiff = (long) (du - dv);
                newEdge += degreeDiff * degreeDiff;
            }

            // Sum up Laplacian values from all neighbor components
            long vLaplacian = 0;
            for (int rv : roots) {
                vLaplacian += currLaplacian[rv];
            }

            // Merge current node with all its processed neighbors into one component
            int s = 0;
            for (int v : uN) {
                s = dsu.union(u.node, v);
            }

            // Update Laplacian for the merged component
            int mergedComponentRoot = dsu.find(u.node);
            currLaplacian[mergedComponentRoot] += vLaplacian + newEdge;

            readded[u.node] = true;

            // Calculate density score
            double sL = s / (currLaplacian[mergedComponentRoot] + epsilon);

            if (sL > sMax) {
                sMax = sL;
                uStar = mergedComponentRoot;
            }
        }

        System.out.println(sMax + ", " + uStar);

        long end = System.nanoTime();
        double elapsedMs = (end - start) / 1_000_000.0;
        System.out.printf("Runtime: %.3f ms%n", elapsedMs);
        return elapsedMs;
    }

    static class DSU {
        public int[] parents = new int[n+1];
        public int[] size = new int[n+1];

        public DSU() {
            for (int i = 1; i < n+1; i++) {
                size[i] = 1;
                parents[i] = i;
            }
        }

        public int find(int x) {
            if (parents[x] !=x) {
                parents[x] = find(parents[x]);
            }
            return parents[x];
        }

        public int union(int n1, int n2) {
            int pn1 = find(n1);
            int pn2 = find(n2);
            if (pn1 == pn2) return 0;
            if (size[pn1] < size[pn2]) {
                int temp = pn1;
                pn1 = pn2;
                pn2 = temp;
            }
            parents[pn2] = pn1;
            size[pn1] += size[pn2];
            return size[pn1];
        }
    }

    static class Pair implements Comparable<Pair> {
        public int node;
        public int degree;

        public Pair(int node, int degree) {
            this.node = node;
            this.degree = degree;
        }

        public int compareTo(Pair o) {
            if (this.degree == o.degree) {
                return Integer.compare(this.node, o.node);
            }
            return Integer.compare(this.degree, o.degree);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair pair = (Pair) o;
            return node == pair.node && degree == pair.degree;
        }

        @Override
        public int hashCode() {
            return Objects.hash(node, degree);
        }
    }
}
