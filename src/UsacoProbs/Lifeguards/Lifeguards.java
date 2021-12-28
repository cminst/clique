package UsacoProbs.Lifeguards;

import java.io.*;
import java.util.HashMap;
import java.util.Scanner;

public class Lifeguards {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("lifeguards.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("lifeguards.out")));

        int n = r.nextInt();
        HashMap<Integer, Integer> lifegaurds = new HashMap<>();

        for (int i = 0; i < n; i++) {
            int starttime = r.nextInt();
            int endtime = r.nextInt();
            lifegaurds.put(starttime, endtime);
        }

        int max = 0;
        for (int i: lifegaurds.keySet()) {
            int[] time = new int[1000];
            int numLifeguard = 0;

            for (int j : lifegaurds.keySet()) {
                if (j != i) {
                    for (int k = j; k < lifegaurds.get(j); k++) {
                        if (time[k] == 0) {
                            numLifeguard++;
                        }
                        time[k] = 1;
                    }
                }
            }

            if (numLifeguard > max) {
                max = numLifeguard;
            }
        }

        pw.println(max);
        pw.close();
    }
}
