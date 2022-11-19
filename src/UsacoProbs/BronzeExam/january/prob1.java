package UsacoProbs.BronzeExam.january;

import java.io.IOException;
import java.util.HashMap;
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
        var hashmap = new HashMap<Character, Integer>();
        for (int i = 0; i < 3; i++) {
            for (char j : answer[i].toCharArray()) {
                if(!hashmap.containsKey(j)) {
                    hashmap.put(j,1);
                }
                else {
                    hashmap.put(j, hashmap.get(j)+1);
                }
            }
        }

        int green = 0;
        int yellow = 0;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if(answer[i].charAt(j)==guess[i].charAt(j)) {
                    green++;
                    hashmap.put(answer[i].charAt(j), hashmap.get(answer[i].charAt(j))-1);
                }
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if(answer[i].charAt(j)!=guess[i].charAt(j) && hashmap.get(guess[i].charAt(j))!=null && hashmap.get(guess[i].charAt(j))!=0) {
                    yellow++;
                    hashmap.put(guess[i].charAt(j), hashmap.get(guess[i].charAt(j))-1);
                }
            }
        }
        System.out.println(green);
        System.out.println(yellow);
    }
}
