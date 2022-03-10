package UsacoProbs.bronze.BlockedBillboard;

import java.io.*;
import java.util.Scanner;

public class BlockedBillboard {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("billboard.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("billboard.out")));
        int[][] billboards = new int[2000][2000];
        for (int i = 0; i < 2; i++) {
            int lowerX = r.nextInt();
            int lowerY = r.nextInt();
            int higherX = r.nextInt();
            int higherY = r.nextInt();
            for (int j = lowerX; j < higherX; j++) {
                for (int k = lowerY; k < higherY; k++) {
                    billboards[j + 1000][k + 1000] = 1;
                }
            }
        }

        int lowerX = r.nextInt();
        int lowerY = r.nextInt();
        int higherX = r.nextInt();
        int higherY = r.nextInt();
        for (int j = lowerX; j < higherX; j++) {
            for (int k = lowerY; k < higherY; k++) {
                billboards[j + 1000][k + 1000] = 2;
            }
        }

        int count = 0;
        for (int i = 0; i < billboards.length; i++) {
            for (int j = 0; j < billboards.length; j++) {
                if (billboards[i][j] == 1) {
                    count++;
                }
            }
        }
        pw.println(count);
        pw.close();
    }
}
