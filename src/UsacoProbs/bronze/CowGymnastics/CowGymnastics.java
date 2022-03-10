package UsacoProbs.bronze.CowGymnastics;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class CowGymnastics {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("gymnastics.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("gymnastics.out")));

        int kSessions = r.nextInt();
        int nCows = r.nextInt();
        ArrayList<HashMap<Integer, Integer>> rankings = new ArrayList<>();
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < kSessions; i++) {
            for (int j = 0; j < nCows; j++) {
                int id = r.nextInt();
                hashMap.put(id, j + 1);
            }
            HashMap<Integer, Integer> h = new HashMap<>(hashMap);
            rankings.add(h);
            hashMap.clear();
        }

        int count = 0;
        for (int pair1 = 1; pair1 <= nCows; pair1++) {
            for (int pair2 = 1; pair2 <= nCows; pair2++) {
                boolean check = true;
                for (int k = 0; k < rankings.size(); k++) {
                    if (rankings.get(k).get(pair1) <= rankings.get(k).get(pair2)) {
                        check = false;
                        k = rankings.size() - 1;
                    }
                }
                if (check) {
                    count++;
                }
            }
        }
        pw.println(count);
        pw.close();
    }
}
