package UsacoProbs.bronze.Photoshoot;

import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public class Photoshoot {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("photo.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("photo.out")));
        int nCows = r.nextInt();
        int[] a = new int[nCows];
        int[] b = new int[nCows - 1];

        for (int i = 0; i < nCows - 1; i++) {
            b[i] = r.nextInt();
        }
        for (int i = 1; i < b[0]; i++) {

            int count = i;
            a[0] = b[0] - i;
            a[1] = i;
            for (int j = 2; j < nCows; j++) {
                int num = b[j - 1];
                int key = num - count;
                if (num - count > 0 && Arrays.stream(a).noneMatch(k -> k == key)) {
                    a[j] = num - count;
                    count = a[j];
                } else {
                    a = new int[nCows];
                    break;
                }
            }
            if (a[0] != 0) {
                break;
            }
        }

        StringBuilder s = new StringBuilder();
        for (int j : a) {
            s.append(j).append(" ");
        }
        s.deleteCharAt(s.length() - 1);
        pw.println(s);
        pw.close();
    }
}
