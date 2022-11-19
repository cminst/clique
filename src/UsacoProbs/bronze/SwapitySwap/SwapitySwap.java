package UsacoProbs.bronze.SwapitySwap;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class SwapitySwap {
    static ArrayList<Integer> cows = new ArrayList<>();
    static int a, a2, b, b2;

    static void reverseAB() {
        int start = a;
        int end = a2;
        while (end - start > 0) {
            Collections.swap(cows, start, end);
            start++;
            end--;
        }
        start = b;
        end = b2;
        while (end - start > 0) {
            Collections.swap(cows, start, end);
            start++;
            end--;
        }
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("swap.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("swap.out")));

        var nCows = r.nextInt();
        var kTimes = r.nextInt();

        for (var i = 0; i < nCows; i++) cows.add(i + 1);

        ArrayList<Integer> originalCows = (ArrayList<Integer>) cows.clone();

        a = r.nextInt() - 1;
        a2 = r.nextInt() - 1;
        b = r.nextInt() - 1;
        b2 = r.nextInt() - 1;
        var count = 0;

        do {
            reverseAB();
            count++;
        } while (!cows.equals(originalCows));

        var m = kTimes % count;

        for (int i = 0; i < m; i++) reverseAB();

        for (var cow : cows) pw.println(cow);
        pw.close();
    }
}
