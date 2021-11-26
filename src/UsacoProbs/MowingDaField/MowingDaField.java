package UsacoProbs.MowingDaField;

import java.io.*;
import java.util.Scanner;

public class MowingDaField {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("mowing.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("mowing.out")));
        int numlines = r.nextInt();
        int[][] arr2d = new int[2001][2001];
        for (int i = 0; i < arr2d.length; i++) {
            for (int j = 0; j < arr2d.length; j++) {
                arr2d[i][j] = -1;
            }
        }
        int x = 1000;
        int y = 1000;
        arr2d[x][y] = 0;
        int t = 0;
        int xgrass = Integer.MAX_VALUE;
        for (int i = 0; i < numlines; i++) {
            String direction = r.next();
            int cells = r.nextInt();
            for (int j = 0; j < cells; j++) {
                if (direction.equals("N")) {
                    y++;
                } else if (direction.equals("E")) {
                    x++;
                } else if (direction.equals("S")) {
                    y--;
                } else {
                    x--;
                }
                if (arr2d[x][y] != -1) {
                    if (t-arr2d[x][y]+1 < xgrass) {
                        xgrass = t-arr2d[x][y]+1;
                    }
                }
                t++;
                arr2d[x][y] = t;
            }
        }
        if (xgrass == Integer.MAX_VALUE) {
            xgrass = -1;
        }
        pw.println(xgrass);
        pw.close();
    }
}
