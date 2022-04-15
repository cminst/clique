package UsacoProbs.silver.LemonadeLine;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class LemonadeLine {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("lemonade.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("lemonade.out")));

        var nCows = r.nextInt();
        var cows = new ArrayList<Integer>();
        for (int i = 0; i < nCows; i++) {
            cows.add(r.nextInt());
        }
        Collections.sort(cows);
        Collections.reverse(cows);

        var count = 0;
        for (Integer cow : cows) {
            if (count <= cow) {
                count++;
            }
        }
        pw.println(count);
        pw.close();
    }
}
