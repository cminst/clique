package UsacoProbs.WordProcessor;

import java.io.*;
import java.util.Scanner;

public class WordProcessor {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("word.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("word.out")));

        int nWords = r.nextInt();
        int maxCharacters = r.nextInt();
        var strings = new String[nWords];
        for (int i = 0; i < nWords; i++) {
            strings[i] = r.next();
        }
        var str = new StringBuilder();
        var numChars = 0;
        var i = 0;
        while (i < strings.length) {
            if (numChars +strings[i].length() <= maxCharacters) {
                str.append(strings[i]);
                numChars+= strings[i].length();
                str.append(" ");
            }
            else {
                str.deleteCharAt(str.length()-1);
                pw.println(str);
                str = new StringBuilder();
                numChars = 0;
                continue;
            }
            i++;
        }
        if (str.length() != 0) {
            str.deleteCharAt(str.length()-1);
            pw.println(str);
        }
        pw.close();
    }
}
