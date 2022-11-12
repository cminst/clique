package UsacoProbs.silver.HighCardWins;

import java.io.*;
import java.util.Scanner;
import java.util.TreeSet;

public class HighCardWins {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("highcard.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("highcard.out")));

        var nCards = r.nextInt();
        var elsieCards = new TreeSet<Integer>();
        var bessieCards = new TreeSet<Integer>();
        for (int i = 0; i < nCards; i++) elsieCards.add(r.nextInt());
        for (int i = 1; i <= nCards * 2; i++) {
            if (!elsieCards.contains(i)) bessieCards.add(i);
        }
        var s = 0;

        var wins = 0;
        for (int i = 0; i < elsieCards.size(); i++) {
            s = elsieCards.higher(s);
            if (bessieCards.higher(s) == null) break;
            wins++;
            bessieCards.remove(bessieCards.higher(s));
        }

        pw.println(wins);
        pw.close();
    }
}
