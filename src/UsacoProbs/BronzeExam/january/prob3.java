package UsacoProbs.BronzeExam.january;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class prob3 {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(System.in);

        int t = r.nextInt();
        Long[] results = new Long[t];
        for (int i = 0; i < t; i++) {
            int n = r.nextInt();
            Long[] hungerLevels = new Long[n];
            for (int j = 0; j < n; j++) hungerLevels[j] = r.nextLong();

            if (n == 1) {
                results[i] = 0L;
                continue;
            } else if (n == 2) {
                results[i] = 0L;
                if (!hungerLevels[0].equals(hungerLevels[1])) results[i] = -1L;
                continue;
            }

            if (hungerLevels[0] > hungerLevels[1]) results[i] = -1L;
            else if (hungerLevels[n - 1] > hungerLevels[n - 2]) results[i] = -1L;
            else {
                long corn = 0;

                for (int k = 1; k < n - 1; k++) {
                    if (hungerLevels[k] > hungerLevels[k - 1]) {
                        long num = hungerLevels[k] - hungerLevels[k - 1];
                        hungerLevels[k] = hungerLevels[k - 1];
                        hungerLevels[k + 1] -= num;
                        corn += num * 2;
                    }
                }

                Collections.reverse(Arrays.asList(hungerLevels));

                for (int k = 1; k < n - 1; k++) {
                    if (hungerLevels[k] > hungerLevels[k - 1]) {
                        while (hungerLevels[k] > hungerLevels[k - 1]) {
                            long num = hungerLevels[k] - hungerLevels[k - 1];
                            hungerLevels[k] = hungerLevels[k - 1];
                            hungerLevels[k + 1] -= num;
                            corn += num * 2;
                        }
                    }
                }

                if (hungerLevels[n - 1] > hungerLevels[n - 2]) {
                    results[i] = -1L;
                    continue;
                }

                if (hungerLevels[1] < 0) results[i] = -1L;
                else results[i] = corn;
            }
        }

        for (long result : results) System.out.println(result);
    }
}
