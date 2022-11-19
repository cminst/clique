package UsacoProbs.bronze.LivestockLineup;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import static java.lang.Math.abs;

public class LivestockLineup {

    static int nDirections;
    static PrintWriter pw = null;
    static ArrayList<String> cowNames = new ArrayList<>(Arrays.asList("Beatrice", "Belinda", "Bella", "Bessie", "Betsy", "Blue", "Buttercup", "Sue"));
    static ArrayList<String>[] requirements = null;
    static boolean result = false;
    static ArrayList<String> list;

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("lineup.in"));
        pw = new PrintWriter(new BufferedWriter(new FileWriter("lineup.out")));
        nDirections = r.nextInt();
        requirements = new ArrayList[2];
        requirements[0] = new ArrayList<>();
        requirements[1] = new ArrayList<>();
        for (int i = 0; i < nDirections; i++) {
            requirements[0].add(r.next());
            for (int j = 0; j < 4; j++) r.next();
            requirements[1].add(r.next());
        }

        permutations(cowNames, new ArrayList<>());
        for (String c : list) pw.println(c);
        pw.close();
    }

    private static boolean check(ArrayList<String> arr) {
        for (int i = 0; i < nDirections; i++) {
            int dist = abs(arr.indexOf(requirements[0].get(i)) - arr.indexOf(requirements[1].get(i)));
            if (dist != 1) return false;
        }
        return true;
    }

    private static void permutations(ArrayList<String> arr, ArrayList<String> k) {
        if (result) return;

        if (arr.size() == 0) {
            if (check(k)) {
                list = k;
                result = true;
                return;
            }
        }

        for (int i = 0; i < arr.size(); i++) {
            String tiredCow = arr.get(i);

            ArrayList<String> notCows = new ArrayList<>(arr);
            notCows.remove(i);

            ArrayList<String> newK = new ArrayList<>(k);
            newK.add(tiredCow);

            permutations(notCows, newK);
        }
    }
}
