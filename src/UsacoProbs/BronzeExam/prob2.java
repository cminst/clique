package UsacoProbs.BronzeExam;

import java.io.IOException;
import java.util.Scanner;

public class prob2 {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(System.in);

        int t = r.nextInt();
        int[][] a = new int[t][4];
        int[][] b = new int[t][4];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < 4; j++) {
                a[i][j] = r.nextInt();
            }
            for (int j = 0; j < 4; j++) {
                b[i][j] = r.nextInt();
            }
        }

        for (int i = 0; i < t; i++) {
            int aPoss = 0;
            int bPoss = 0;
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (a[i][j] > b[i][k]) {
                        aPoss++;
                    }
                    if (a[i][j] < b[i][k]) {
                        bPoss++;
                    }
                }
            }
            boolean BbeatsA = false;

            if (bPoss > aPoss) {
                BbeatsA = true;
            }

            boolean check = false;
            int[] c = new int[4];
            for (int j = 1; j < 11; j++) {
                c[0] = j;
                for (int k = 1; k < 11; k++) {
                    c[1] = k;
                    for (int l = 1; l < 11; l++) {
                        c[2] = l;
                        for (int m = 1; m < 11; m++) {
                            c[3] = m;
                            aPoss = 0;
                            int cPoss = 0;
                            for (int n = 0; n < 4; n++) {
                                for (int o = 0; o < 4; o++) {
                                    if (a[i][n] > c[o]) {
                                        aPoss++;
                                    }
                                    if (a[i][n] < c[o]) {
                                        cPoss++;
                                    }
                                }
                            }
                            bPoss = 0;
                            int cPoss2 = 0;
                            for (int n = 0; n < 4; n++) {
                                for (int o = 0; o < 4; o++) {
                                    if (c[o] > b[i][n]) {
                                        cPoss2++;
                                    }
                                    if (c[o] < b[i][n]) {
                                        bPoss++;
                                    }
                                }
                            }

                            if ((BbeatsA && cPoss2 > bPoss && aPoss > cPoss) || (!BbeatsA && cPoss2 < bPoss && aPoss < cPoss)) {
                                System.out.println("yes");
                                check = true;
                                break;
                            }
                        }
                        if (check) {
                            break;
                        }
                    }
                    if (check) {
                        break;
                    }
                }
                if (check) {
                    break;
                }
            }
            if (!check) {
                System.out.println("no");
            }
        }
    }
}
