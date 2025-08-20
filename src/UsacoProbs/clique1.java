package UsacoProbs;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class clique1 {
    public static int n;
    public static double main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader(args[0]));

        n = r.nextInt();
        int m = r.nextInt();

        long start = System.nanoTime();

        TreeMap<Integer, ArrayList<Integer>> map = new TreeMap<>();
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        int[] degrees = new int[n+1];
        Stack<Pair> stack = new Stack<>();

        for (int i = 0; i < n; i++) {
            map.put(i+1, new ArrayList<>());
        }
        for (int i = 0; i < m; i++) {
            int node1 = r.nextInt();
            int node2 = r.nextInt();
            map.get(node1).add(node2);
            map.get(node2).add(node1);
        }

        for (int i = 1; i < n+1; i++) {
            pq.add(new Pair(i, map.get(i).size()));
            degrees[i] = map.get(i).size();
        }

        while(!pq.isEmpty()) {
            Pair minDegreeNode = pq.poll();
            if (degrees[minDegreeNode.node]!=minDegreeNode.degree) continue;
            for (int connectedNode : map.get(minDegreeNode.node)) {
                if (degrees[connectedNode]!=0) {
                    degrees[connectedNode] -= 1;
                    pq.add(new Pair(connectedNode, degrees[connectedNode]));
                }
            }
            degrees[minDegreeNode.node] = 0;
            stack.add(minDegreeNode);
        }
        DSU dsu = new DSU();
        int sMax = 0;
        int uStar = 0;

        boolean[] readded = new boolean[n+1];
        while (!stack.isEmpty()) {
            Pair minDegreeNode = stack.pop();
            for (int connectedNode : map.get(minDegreeNode.node)) {
                if (readded[connectedNode]) {
                    int s = dsu.union(minDegreeNode.node, connectedNode);
                    if (s!=0) {
                        int newScore = s*minDegreeNode.degree;
                        if (newScore>sMax) {
                            sMax = newScore;
                            uStar = dsu.find(minDegreeNode.node);
                        }
                    }
                }
            }
            readded[minDegreeNode.node] = true;
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