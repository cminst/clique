package UsacoProbs.WhereAm1;

import java.io.*;
import java.util.HashMap;
import java.util.Scanner;

public class WhereAm1 {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("whereami.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("whereami.out")));

        int nMailboxes = r.nextInt();
        String str = r.next();
        HashMap<String, Integer> mailboxes = new HashMap<>();
        for (int i = 0; i < nMailboxes; i++) {
            mailboxes.merge(String.valueOf(str.charAt(i)), 1, Integer::sum);
        }

        int max = 0;
        int notDuplicates = 0;
        for (int i = 0; i < nMailboxes + 1; i++) {
            for (int j = i; j < nMailboxes; j++) {
                String s = str.substring(i, j+1);
                int count = 0;
                int first = 0;
                while (first <= str.length()-s.length()) {
                    int second = first+s.length();
                    if (str.substring(first, second).equals(s)) {
                        count++;
                    }
                    first++;
                }

                notDuplicates++;
                if (count == 1) {
                    if (notDuplicates > max) {
                        max = notDuplicates;
                    }
                    notDuplicates = 0;
                    break;
                }
            }
        }
        pw.println(max);
        pw.close();
    }
}
