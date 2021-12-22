package UsacoProbs.MilkPails;

import java.io.*;
import java.util.Scanner;

public class MilkPails {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("pails.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("pails.out")));

        int x = r.nextInt();
        int y = r.nextInt();
        int m = r.nextInt();
        int max = 0;
        int numY = 0;
        if (m/y > max) {
            max = m/y;
        }
        int num = m/y;
        for (int i = 0; i <= num; i++) {
            if (x*(m/x)+(numY*y) > max) {
                max = x*(m/x)+(numY*y);
            }
            m -= y;
            numY++;
        }
        pw.println(max);
        pw.close();
    }
}
