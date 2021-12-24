package UsacoProbs.Lifeguards;

import java.io.*;
import java.util.HashMap;
import java.util.Scanner;

public class Lifeguards {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("lifeguards.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("lifeguards.out")));

        int nLifeguards = r.nextInt();
        HashMap<Integer, Integer> pairs = new HashMap<>();

        for (int i = 0; i < nLifeguards; i++) {
            int start = r.nextInt();
            int end = r.nextInt();
            pairs.put(start, end);
        }

        int maxLifeguardTime = 0;
        for (int i : pairs.keySet()) {
            int[] shifts = new int[1000];
            int lifeguardTime = 0;

            for (int j : pairs.keySet()) {
                if (j != i) {
                    for (int k = j; k < pairs.get(j); k++) {
                        if (shifts[k] == 0){
                            lifeguardTime++;
                        }
                        shifts[k] += 1;
                    }
                }
            }

            if (lifeguardTime > maxLifeguardTime) {
                maxLifeguardTime = lifeguardTime;
            }
        }
        pw.println(maxLifeguardTime);
        pw.close();
    }
}
