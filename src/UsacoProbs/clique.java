package UsacoProbs;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class clique {
    public static int n;
    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("src/UsacoProbs/input.txt"));

        n = r.nextInt();
        int m = r.nextInt();

        TreeMap<Integer, ArrayList<Integer>> map = new TreeMap<>();
        for (int i = 0; i < n; i++) {
            map.put(i+1, new ArrayList<>());
        }
        for (int i = 0; i < m; i++) {
            int node1 = r.nextInt();
            int node2 = r.nextInt();
            map.get(node1).add(node2);
            map.get(node2).add(node1);
        }

        PriorityQueue<Pair> pq = new PriorityQueue<>();
//        ArrayList<Integer> pqTest = new ArrayList<>();
        int[] degrees = new int[n+1];

        for (int i = 1; i < n+1; i++) {
            pq.add(new Pair(i, map.get(i).size()));
            degrees[i] = map.get(i).size();
        }

        Stack<Pair> stack = new Stack<>();
//        ArrayList<Integer> stackTest = new ArrayList<>();
        while(!pq.isEmpty()) {
            Pair minDegreeNode = pq.poll();
            for (int connectedNode : map.get(minDegreeNode.node)) {
                if (!stack.contains(new Pair(connectedNode, degrees[connectedNode]))) {
                    pq.remove(new Pair(connectedNode, degrees[connectedNode]));
                    degrees[connectedNode] -= 1;
                    pq.add(new Pair(connectedNode, degrees[connectedNode]));
                }
            }
            stack.add(minDegreeNode);
//            stackTest.add(minDegreeNode.node);
        }
//        System.out.println(stackTest);
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
//                        System.out.println(s+" "+minDegreeNode.node);
                        int newScore = s*minDegreeNode.degree;
                        if (newScore>sMax) {
                            sMax = newScore;
                            uStar = minDegreeNode.node;
                        }
                    }
                }
            }
            readded[minDegreeNode.node] = true;
        }

        System.out.println(sMax + ", " + uStar);
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
}