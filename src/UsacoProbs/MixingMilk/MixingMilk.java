package UsacoProbs.MixingMilk;

import java.io.*;
import java.util.Scanner;

public class MixingMilk {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("mixmilk.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("mixmilk.out")));
        int aCapacity = r.nextInt();
        int aMilk = r.nextInt();
        int bCapacity = r.nextInt();
        int bMilk = r.nextInt();
        int cCapacity = r.nextInt();
        int cMilk = r.nextInt();
        int[] capacity = new int[]{aCapacity, bCapacity, cCapacity};
        int[] milk = new int[]{aMilk, bMilk, cMilk};

        for (int i = 0, j = 0; j < 100; j++, i++) {
            if (i == 3) {
                i = 0;
            }
            int next = i + 1;
            if (i == 2) {
                next = 0;
            }
            int amt = Math.min(milk[i], capacity[next] - milk[next]);
            milk[i] -= amt;
            milk[next] += amt;
        }

        for (int i = 0; i < 3; i++) {
            pw.println(milk[i]);
        }
        pw.close();

    }
}
