package UsacoProbs.BronzeExam;

import java.util.Scanner;

public class RecordMilking {

    public static void main(String[] args) {
        Scanner r = new Scanner(System.in);
        int recordMilking = r.nextInt();
        int bessieMilking = r.nextInt();
        int[] recordRates = new int[100];
        int[] bessieRates = new int[100];
        int index = 0;
        for (int i = 0; i < recordMilking; i++) {
            int recordMinutes = r.nextInt();
            int recordRate = r.nextInt();
            int ind = index;
            for (int j = 0; j < recordMinutes; j++) {
                recordRates[j + ind] = recordRate;
                index++;
            }
        }

        index = 0;
        for (int i = 0; i < bessieMilking; i++) {
            int bessieMinutes = r.nextInt();
            int bessieRate = r.nextInt();
            int ind = index;
            for (int j = 0; j < bessieMinutes; j++) {
                bessieRates[j + ind] = bessieRate;
                index++;
            }
        }

        int maxDiff = 0;
        for (int i = 0; i < bessieRates.length; i++) {
            if (bessieRates[i] - recordRates[i] > maxDiff) {
                maxDiff = bessieRates[i] - recordRates[i];
            }
        }
        System.out.println(maxDiff);
    }
}