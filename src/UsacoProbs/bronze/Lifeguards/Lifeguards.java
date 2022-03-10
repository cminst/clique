package UsacoProbs.bronze.Lifeguards;

import java.io.*;
import java.util.HashMap;
import java.util.Scanner;

public class Lifeguards {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("lifeguards.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("lifeguards.out")));

        int n = r.nextInt();
        HashMap<Integer, Integer> lifeguards = new HashMap<>();

        for (int i = 0; i < n; i++) {
            int startTime = r.nextInt();
            int endTime = r.nextInt();
            lifeguards.put(startTime, endTime);
        }

        int max = 0;
        for (int i : lifeguards.keySet()) {
            int[] time = new int[1000];
            int numLifeguard = 0;

            for (int j : lifeguards.keySet()) {
                if (j != i) {
                    for (int k = j; k < lifeguards.get(j); k++) {
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
