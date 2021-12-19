package UsacoProbs.Triangles;

import java.io.*;
import java.util.Scanner;

public class Triangles {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("triangles.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("triangles.out")));
        int nPoints = r.nextInt();
        int[] x = new int[nPoints];
        int[] y = new int[nPoints];

        for (int i = 0; i < nPoints; i++) {
            x[i] = r.nextInt();
            y[i] = r.nextInt();
        }

        int result = 0;
        for (int i = 0; i < nPoints; i++) {
            for (int xCurrent = 0; xCurrent < nPoints; xCurrent++) {
                if (i == xCurrent || x[i] != x[xCurrent])
                    continue;
                for (int yCurrent = 0; yCurrent < nPoints; yCurrent++) {
                    if (i == yCurrent || y[i] != y[yCurrent])
                        continue;
                    int area = Math.abs(x[yCurrent] - x[i]) * Math.abs(y[xCurrent] - y[i]);
                    if (area > result)
                        result = area;
                }
            }
        }

        pw.println(result);
        pw.close();
    }
}