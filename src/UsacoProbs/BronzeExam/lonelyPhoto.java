package UsacoProbs.BronzeExam;

import java.io.IOException;
import java.util.Scanner;

public class lonelyPhoto {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(System.in);

        int nCows = r.nextInt();
        String str = r.next();

        int count = 0;
        int i = 0;
        while (i < nCows) {

            int right = nCows - (i + 1);
            if (right >= 1) {
                for (int j = 2; j <= right; j++) {
                    int h = (int) str.substring(i, i + j + 1).chars().filter(f -> f == 'H').count();
                    int g = (int) str.substring(i, i + j + 1).chars().filter(f -> f == 'G').count();
                    if (g == 1 || h == 1) {
                        count++;
                    }
                }
            }
            i++;
        }
        System.out.println(count);
    }
}
