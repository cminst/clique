package UsacoProbs.bronze.BlockedBillboard2;

import java.io.*;
import java.util.Scanner;

public class BlockedBillboard2 {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("billboard2.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("billboard2.out")));

        int[][] billboards = new int[2000][2000];

        int lowerX = r.nextInt();
        int lowerY = r.nextInt();
        int higherX = r.nextInt();
        int higherY = r.nextInt();
        for (int i = lowerX; i < higherX; i++) {
            for (int j = lowerY; j < higherY; j++) {
                billboards[i + 1000][j + 1000] = 1;
            }
        }

        lowerX = r.nextInt();
        lowerY = r.nextInt();
        higherX = r.nextInt();
        higherY = r.nextInt();
        for (int i = lowerX; i < higherX; i++) {
            for (int j = lowerY; j < higherY; j++) {
                billboards[i + 1000][j + 1000] = 0;
            }
        }

        int minX = Integer.MAX_VALUE;
        int minY = Integer.MAX_VALUE;
        int maxX = Integer.MIN_VALUE;
        int maxY = Integer.MIN_VALUE;
        for (int i = 0; i < billboards.length; i++) {
            for (int j = 0; j < billboards.length; j++) {
                if (billboards[i][j] == 1) {
                    if (i < minX) {
                        minX = i;
                    }

                    if (j < minY) {
                        minY = j;
                    }

                    if (i > maxX) {
                        maxX = i;
                    }

                    if (j > maxY) {
                        maxY = j;
                    }
                }
            }
        }

        if (maxX == Integer.MIN_VALUE && maxY == Integer.MIN_VALUE && minX == Integer.MAX_VALUE && minY == Integer.MAX_VALUE) {
            pw.println(0);
            pw.close();
        }
        pw.println((maxX + 1 - minX) * (maxY + 1 - minY));
        pw.close();
    }
}
