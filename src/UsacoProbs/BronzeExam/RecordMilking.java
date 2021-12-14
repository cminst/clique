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
                recordRates[j+ind] = recordRate;
                index++;
            }
        }

        index = 0;
        for (int i = 0; i < bessieMilking; i++) {
            int bessieMinutes = r.nextInt();
            int bessieRate = r.nextInt();
            int ind = index;
            for (int j = 0; j < bessieMinutes; j++) {
                bessieRates[j+ind] = bessieRate;
                index++;
            }
        }

        int maxDiff = 0;
        for (int i = 0; i < bessieRates.length; i++) {
            if (bessieRates[i]-recordRates[i] > maxDiff) {
                maxDiff = bessieRates[i]-recordRates[i];
            }
        }
        System.out.println(maxDiff);
    }
}

// import java.util.Scanner;
//
//public class RecordMilking {
//
//    public static void main(String[] args) {
//        Scanner r = new Scanner(System.in);
//        int recordMilking = r.nextInt();
//        int bessieMilking = r.nextInt();
//        int[] recordRates = new int[recordMilking];
//        int[] bessieRates = new int[bessieMilking];
//        for (int i = 0; i < recordMilking; i++) {
//            r.nextInt();
//            int recordRate = r.nextInt();
//            recordRates[i] = recordRate;
//        }
//
//        for (int i = 0; i < bessieMilking; i++) {
//            r.nextInt();
//            int bessieRate = r.nextInt();
//            bessieRates[i] = bessieRate;
//        }
//
//        int rateChange = 0;
//        for (int i = 0; i < Math.min(recordRates.length, bessieRates.length); i++) {
//            rateChange = Math.max(bessieRates[i]-recordRates[i], rateChange);
//        }
//        System.out.println(rateChange);
//    }
//}