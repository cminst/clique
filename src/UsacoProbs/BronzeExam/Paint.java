package UsacoProbs.BronzeExam;

import java.io.IOException;
import java.util.Scanner;

public class Paint {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(System.in);
        int farmerStart = r.nextInt();
        int farmerEnd = r.nextInt();
        int bessieStart = r.nextInt();
        int bessieEnd = r.nextInt();
        int[] fence = new int[101];

        for (int i = 0; i < farmerEnd - farmerStart; i++) {
            fence[i+farmerStart] = 1;
        }

        for (int i = 0; i < bessieEnd - bessieStart; i++) {
            fence[i+bessieStart] = 1;
        }

        int count = 0;
        for (int i = 0; i < 101; i++) {
            if (fence[i]  == 1) {
                count++;
            }
        }
        System.out.println(count);
    }
}
