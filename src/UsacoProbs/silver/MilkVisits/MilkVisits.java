package UsacoProbs.silver.MilkVisits;

import java.io.*;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.TreeSet;

public class MilkVisits {
    static ArrayList<ArrayList<Node>> adj = new ArrayList<>();
    static String str;
    static int[] types;
    static boolean[] visited;

    public static void main(String[] args) throws IOException {
        BufferedReader r = new BufferedReader(new FileReader("milkvisits.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("milkvisits.out")));

        var st = new StringTokenizer(r.readLine());
        var n = Integer.parseInt(st.nextToken());
        var m = Integer.parseInt(st.nextToken());

        TreeSet<Integer> roots = new TreeSet<>();
        for (int i = 0; i < n; i++) {
            adj.add(new ArrayList<>());
            roots.add(i);
        }

        str = r.readLine();
        types = new int[n];
        visited = new boolean[n];
        for (int i = 0; i < n - 1; i++) {
            st = new StringTokenizer(r.readLine());
            var first = Integer.parseInt(st.nextToken()) - 1;
            var second = Integer.parseInt(st.nextToken()) - 1;
            adj.get(first).add(new Node(second));
            roots.remove(second);
        }
        shorten(roots.first(), roots.first());

        for (int i = 0; i < m; i++) {
            st = new StringTokenizer(r.readLine());
            var start = Integer.parseInt(st.nextToken()) - 1;
            var end = Integer.parseInt(st.nextToken()) - 1;
            var type = st.nextToken();
            if (types[start] != types[end] || String.valueOf(str.charAt(start)).equals(type) || String.valueOf(str.charAt(end)).equals(type)) {
                pw.print(1);
            } else {
                pw.print(0);
            }
        }
        pw.close();
    }

    static void shorten(int barn, int firstSameType) {
        if (!visited[barn]) {
            if (str.charAt(firstSameType) != str.charAt(barn)) {
                firstSameType = barn;
            }

            types[barn] = firstSameType;
            for (Node next : adj.get(barn)) {
                int nextSameType = firstSameType;
                if (next.milk != str.charAt(firstSameType)) {
                    nextSameType = next.val;
                }
                shorten(next.val, nextSameType);
            }
            visited[barn] = true;
        }
    }

    static class Node {
        int val;
        char milk;

        public Node(int i) {
            val = i;
            milk = str.charAt(i);
        }
    }
}