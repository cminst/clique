package UsacoProbs.bronze.BlockGame;

import java.io.*;
import java.util.HashMap;
import java.util.Scanner;

public class BlockGame {

    public static void addOccurrences(HashMap<Character, Integer> hashmap, Object c) {
        if (hashmap.containsKey(c.toString().toCharArray()[0])) hashmap.put((Character) c, hashmap.get(c) + 1);
        else hashmap.put((Character) c, 1);
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("blocks.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("blocks.out")));

        String[] alphabet = new String[]{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};
        int nSets = r.nextInt();
        var first = new String[nSets];
        var second = new String[nSets];
        for (int i = 0; i < nSets; i++) {
            first[i] = r.next();
            second[i] = r.next();
        }
        var result = new int[26];

        for (int i = 0; i < 26; i++) {
            var c = alphabet[i];
            var maxCount = 0;
            for (int j = 0; j < first.length; j++) {
                int count = first[j].length() - first[j].replace(c, "").length();
                int count2 = second[j].length() - second[j].replace(c, "").length();
                maxCount += Math.max(count, count2);
            }
            result[i] = maxCount;
        }

        for (int j : result) pw.println(j);
        pw.close();
    }
}
