package UsacoProbs.BronzeExam;

import java.util.Scanner;

public class MilkSchedule {

    public static void main(String[] args) {
        Scanner r = new Scanner(System.in);

        int nCows = r.nextInt();
        int qQuestions = r.nextInt();

        int[] cows = new int[nCows];
        int[] questions = new int[qQuestions];
        int[] time = new int[nCows + 1];
        time[0] = 0;
        for (int i = 0; i < nCows; i++) {
            cows[i] = r.nextInt();
            time[i + 1] = time[i] + cows[i];
        }
        for (int i = 0; i < qQuestions; i++) {
            questions[i] = r.nextInt();
        }

        for (int i = 0; i < qQuestions; i++) {
            int cowIndex = questions[i];
            for (int j = 0; j < time.length; j++) {
                if (time[j] > cowIndex) {
                    if (time[j - 1] <= cowIndex) {
                        System.out.println(j);
                        break;
                    }
                }
                if (time[j] == cowIndex) {
                    System.out.println(j + 1);
                    break;
                }
            }
        }
    }
}
