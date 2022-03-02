package UsacoProbs.BronzeExam.january;

import java.io.IOException;
import java.util.Scanner;

public class prob1 {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(System.in);

        String[] answer = new String[3];
        for (int i = 0; i < 3; i++) {
            String n = r.next();
            answer[i] = n;
        }
        String[] guess = new String[3];
        for (int i = 0; i < 3; i++) {
            String n = r.next();
            guess[i] = n;
        }

        int green = 0;
        int yellow = 0;
        for (int i = 0; i < answer.length; i++) {
            for (int j = 0; j < answer.length; j++) {
                if (answer[i].charAt(j) == guess[i].charAt(j)) {
                    green++;
                    StringBuilder str = new StringBuilder(guess[i]);
                    str.setCharAt(j, '1');
                    guess[i] = str.toString();
                    str = new StringBuilder(answer[i]);
                    str.setCharAt(j, '1');
                    answer[i] = str.toString();
                }
            }
        }

        for (int i = 0; i < answer.length; i++) {
            for (int j = 0; j < answer.length; j++) {
                if (answer[i].charAt(j) != guess[i].charAt(j) && (guess[0].contains(Character.toString(answer[i].charAt(j))) || guess[1].contains(Character.toString(answer[i].charAt(j))) || guess[2].contains(Character.toString(answer[i].charAt(j))))) {
                    yellow++;
                    var inum = 0;
                    if (guess[1].contains(Character.toString(answer[i].charAt(j)))) {
                        inum = 1;
                    }
                    if (guess[2].contains(Character.toString(answer[i].charAt(j)))) {
                        inum = 2;
                    }
                    var jnum = 0;
                    if (guess[inum].charAt(1) == answer[i].charAt(j)) {
                        jnum = 1;
                    }
                    if (guess[inum].charAt(2) == answer[i].charAt(j)) {
                        jnum = 2;
                    }
                    StringBuilder str = new StringBuilder(guess[inum]);
                    str.setCharAt(jnum, '1');
                    guess[inum] = str.toString();
                    str = new StringBuilder(answer[i]);
                    str.setCharAt(j, '1');
                    answer[i] = str.toString();
                }
            }
        }
        System.out.println(green);
        System.out.println(yellow);
    }
}
