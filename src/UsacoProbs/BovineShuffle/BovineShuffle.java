package UsacoProbs.BovineShuffle;

import java.io.*;
import java.util.Scanner;

public class BovineShuffle {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("shuffle.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("shuffle.out")));
        int numCows = r.nextInt();
        int[] shuffle = new int[numCows];
        int[] cowIDs = new int[numCows];
        for (int i = 0; i < numCows; i++)
            shuffle[i] = r.nextInt() - 1;
        for (int i = 0; i < numCows; i++)
            cowIDs[i] = r.nextInt();
        int[] temparr = cowIDs.clone();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < numCows; j++) {
                temparr[j] = cowIDs[shuffle[j]];
            }
            cowIDs = temparr.clone();
        }
        for (int i = 0; i < numCows; i++) {
            pw.println(cowIDs[i]);
        }
        pw.close();
    }
}
