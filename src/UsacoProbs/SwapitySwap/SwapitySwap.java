package UsacoProbs.SwapitySwap;

import java.io.*;
import java.util.Scanner;

public class SwapitySwap {
    static int[] cows;

    public static void swap(int firstIndex, int secondIndex) {
        var firstCow = cows[firstIndex];
        cows[firstIndex] = cows[secondIndex];
        cows[secondIndex] = firstCow;
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("swap.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("swap.out")));

        int nCows = r.nextInt();
        int kTimes = r.nextInt();

        cows = new int[nCows];
        for (int i = 0; i < nCows; i++) {
            cows[i] = i+1;
        }

        int a = r.nextInt();
        int a2 = r.nextInt();
        int b = r.nextInt();
        int b2 = r.nextInt();

        for (int i = 0; i < kTimes; i++) {
            for (int j = a; j < (a2 - (a - 1)) / 2+a; j++) {
                swap(j-1, a2 - (j - a)-1);
            }
            for (int j = b; j < (b2 - (b - 1)) / 2+b; j++) {
                swap(j-1, b2 - (j - b)-1);
            }
        }

        for (int cow : cows) {
            pw.println(cow);
        }
        pw.close();
    }
}
