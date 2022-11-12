package UsacoProbs.gold;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.StringTokenizer;

public class ClosingDaFarm {
    static ArrayList<ArrayList<Integer>> adj = new ArrayList<>();
    static ArrayList<Integer> open = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        BufferedReader r = new BufferedReader(new FileReader("closing.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("closing.out")));
        StringTokenizer st = new StringTokenizer(r.readLine());
        var n = Integer.parseInt(st.nextToken());
        var m = Integer.parseInt(st.nextToken());
        for (int i = 0; i < n; i++) {
            adj.add(new ArrayList<>());
            open.add(i);
        }
        for (int i = 0; i < m; i++) {
            st = new StringTokenizer(r.readLine());
            var first = Integer.parseInt(st.nextToken()) - 1;
            var second = Integer.parseInt(st.nextToken()) - 1;
            adj.get(first).add(second);
            adj.get(second).add(first);
        }
        var start = open.get(0);
        if (getSize(start, new HashSet<>()) != n) pw.println("NO");
        else pw.println("YES");

        for (int i = 1; i < n - 1; i++) {
            var del = Integer.parseInt(r.readLine()) - 1;
            open.remove((Object) del);
            for (int j : adj.get(del)) adj.get(j).remove((Object) del);
            start = open.get(0);
            adj.get(del).clear();
            if (getSize(start, new HashSet<>()) != n - i) pw.println("NO");
            else pw.println("YES");
        }
        pw.println("YES");
        pw.close();
    }

    private static int getSize(int barn, HashSet<Integer> visited) {
        for (int i : adj.get(barn)) {
            if (visited.add(i)) getSize(i, visited);
        }
        return visited.size();
    }
}
