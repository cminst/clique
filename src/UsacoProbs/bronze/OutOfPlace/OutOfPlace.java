package UsacoProbs.bronze.OutOfPlace;

import java.io.*;
import java.util.HashSet;
import java.util.Scanner;

public class OutOfPlace {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("outofplace.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("outofplace.out")));

        int nCows = r.nextInt();
        var cows = new int[nCows + 1];
        cows[1] = r.nextInt();
        var bessie = cows.length - 1;
        var actualSpot = 0;
        for (int i = 2; i < nCows + 1; i++) {
            cows[i] = r.nextInt();
        }

        for (int i = 1; i < nCows - 1; i++) {
            if (cows[i] < cows[i - 1]) {
                if (cows[i + 1] >= cows[i - 1]) {
                    bessie = i;
                } else {
                    bessie = i - 1;
                }
            }
        }

        for (int i = 0; i < cows.length; i++) {
            if (cows[i] < cows[bessie]) {
                actualSpot = i + 1;
            }
        }

        var hashset = new HashSet<Integer>();
        if (bessie > actualSpot) {
            if (cows[actualSpot] == cows[bessie]) actualSpot++;
            for (int i = actualSpot; i < bessie; i++) hashset.add(cows[i]);
        } else {
            if (cows[actualSpot] < cows[bessie]) actualSpot++;
            for (int i = bessie + 1; i < actualSpot; i++) hashset.add(cows[i]);
        }

        pw.println(hashset.size());
        pw.close();
    }
}
