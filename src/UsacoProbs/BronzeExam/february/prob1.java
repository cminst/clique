package UsacoProbs.BronzeExam.february;

import java.util.Scanner;

public class prob1 {

    public static void main(String[] args) {
        Scanner r = new Scanner(System.in);

        int t = r.nextInt();
        for (int i = 0; i < t; i++) {
            int nClasses = r.nextInt();
            int[] naps = new int[nClasses];
            var original = new int[nClasses];
            naps[0] = r.nextInt();
            original[0] = naps[0];
            var allSame = true;
            var nap = naps[0];
            var max = 0;

            var sumOfNaps = naps[0];
            for (int j = 1; j < nClasses; j++) {
                var time = r.nextInt();
                naps[j] = time;
                original[j] = time;
                if (naps[j] != nap) {
                    allSame = false;
                }
                if (naps[j] > max) {
                    max = naps[j];
                }
                sumOfNaps += naps[j];
            }

            if (allSame) {
                System.out.println(0);
                continue;
            }
            var modifications = 0;
            var check = false;
            var target = 0;
            for (int j = max; j <= sumOfNaps; j++) {
                if (sumOfNaps%j == 0) {
                    target = j;
                    check= false;
                    for (int k = 0; k < naps.length-1; k++) {
                        if (naps[k] < target) {
                            naps[k+1] += naps[k];
                            modifications++;
                        }
                        if (naps[k] > target) {
                            check = true;
                            var o = original.clone();
                            naps = o;
                            modifications = 0;
                            break;
                        }
                    }
                    if (!check) {
                        break;
                    }
                }
            }
            if (!check) {
                System.out.println(modifications);
            }
        }
    }
}
/*
1
7
6 6 6 3 3 6 6

1
7
9 9 4 1 4 9 9

1
26
4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 100

1
8
2 3 2 3 3 3 2 2

1
7
3 2 2 2 3 1 2

1
5
2 2 2 3 3

1
8
1 1 1 1 1 1 2 1

1
6
1 2 3 1 1 1

1
3
2 2 3

1
5
0 0 0 0 0

1
7
3 2 1 1 2 2 1

1
6
8 4 4 8 4 4

1
3
16 1 1

1
1
13

 */