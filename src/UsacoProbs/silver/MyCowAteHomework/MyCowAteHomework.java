package UsacoProbs.silver.MyCowAteHomework;

import java.io.*;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class MyCowAteHomework {

    public static void main(String[] args) throws IOException {
        BufferedReader r = new BufferedReader(new FileReader("homework.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("homework.out")));

        var st = new StringTokenizer(r.readLine());
        var n = Integer.parseInt(st.nextToken());
        var arr = new int[n];
        var prefix = new int[n];
        var mins = new int[n];
        st = new StringTokenizer(r.readLine());
        for (int i = 0; i < n; i++) {
            arr[i] = Integer.parseInt(st.nextToken());
        }

        prefix[n-1] = arr[n-1];
        for (int i = n-2; i >=0; i--) {
            prefix[i] = prefix[i + 1] + arr[i];
        }
        var min = Integer.MAX_VALUE;
        for (int i = n - 1; i >= 0; i--) {
            min = Math.min(min, arr[i]);
            mins[i] = min;
        }
        var maxes = new ArrayList<Integer>();
        var max = 0.0;
        for (int i = 1; i < n - 1; i++) {
            var average = (double) (prefix[i] - mins[i]) / (arr.length - i - 1);
            if (average > max) {
                maxes = new ArrayList<>();
                max = average;
            } else if (average != max) {
                continue;
            }
            maxes.add(i);
        }
        for (int i : maxes) {
            pw.println(i);
        }
        pw.close();
    }
}