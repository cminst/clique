package UsacoProbs.BronzeExam.open23;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class MooLanguage {
    public static void main(String[] args) throws IOException {
        BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
        var st = new StringTokenizer(r.readLine());

        var testcases = Integer.parseInt(st.nextToken());
        for (int q = 0; q < testcases; q++) {
            var nouns = new ArrayList<String>();
            var inVerbs = new ArrayList<String>();
            var transVerbs = new ArrayList<String>();
            var conjunctions = new ArrayList<String>();
            st = new StringTokenizer(r.readLine());
            var words = Integer.parseInt(st.nextToken());
            var commas = Integer.parseInt(st.nextToken());
            var periods = Integer.parseInt(st.nextToken());
            for (int w = 0; w < words; w++) {
                st = new StringTokenizer(r.readLine());
                var word = st.nextToken();
                var type = st.nextToken();
                if (type.charAt(0) == 'n') nouns.add(word);
                else if (type.charAt(0) == 'i') inVerbs.add(word);
                else if (type.charAt(0) == 't') transVerbs.add(word);
                else conjunctions.add(word);
            }
            var n = nouns.size();
            var i = inVerbs.size();
            var t = transVerbs.size();
            var c = conjunctions.size();
            var maxWords = 0;
            var maxStr = new StringBuilder();

            for (int j = 0; j <= Math.min(t, n / 2); j++) {
                var type2 = j;
                var type1 = Math.min(i, n - type2 * 2);
                var groups = Math.min(c, (type1 + type2) / 2);
                var numSentences = type1 + type2 - groups;
                var notJoined = (type1 + type2) - groups * 2;
                var remove = 0;
                if (numSentences > periods) {
                    var removeSentences = numSentences - periods;
                    if (notJoined >= removeSentences) remove = removeSentences;
                    else remove = notJoined + (removeSentences - notJoined) * 2;
                }
                if (remove <= type1) type1 -= remove;
                else {
                    type2 -= remove - type1;
                    type1 = 0;
                }
                groups = Math.min(c, (type1 + type2) / 2);
                var addedNouns = 0;
                if (type2 > 0) {
                    addedNouns = Math.min(commas, n - type1 - type2 * 2);
                }
                var numWords = type2 * 3 + type1 * 2 + groups + addedNouns;
                if (numWords > maxWords) {
                    maxWords = numWords;
                    var str = new StringBuilder();
                    var firstSentence = true;
                    var k = 0;
                    while (k < type2) {
                        if (!firstSentence) str.append(" ");
                        str.append("n").append(" t").append(" n");
                        k++;
                        while (addedNouns > 0) {
                            str.append(", n");
                            addedNouns--;
                        }
                        if (groups > 0) {
                            groups--;
                            if (k != type2) {
                                str.append(" c").append(" n").append(" t").append(" n.");
                                k++;
                            } else if (type1 > 0) {
                                str.append(" c").append(" n").append(" i.");
                                type1--;
                            }
                        } else str.append(".");
                        firstSentence = false;
                    }

                    k = 0;
                    while (k < type1) {
                        if (!firstSentence) str.append(" ");
                        str.append("n").append(" i");
                        k++;
                        if (groups > 0 && k != type1) {
                            groups--;
                            str.append(" c").append(" n").append(" i.");
                            k++;
                        } else str.append(".");
                        firstSentence = false;
                    }
                    maxStr = str;
                }
            }

            System.out.println(maxWords);

            var nIndex = 0;
            var iIndex = 0;
            var tIndex = 0;
            var cIndex = 0;
            for (char ch : maxStr.toString().toCharArray()) {
                if (ch == 'n') {
                    System.out.print(nouns.get(nIndex));
                    nIndex++;
                } else if (ch == 'i') {
                    System.out.print(inVerbs.get(iIndex));
                    iIndex++;
                } else if (ch == 't') {
                    System.out.print(transVerbs.get(tIndex));
                    tIndex++;
                } else if (ch == 'c') {
                    System.out.print(conjunctions.get(cIndex));
                    cIndex++;
                } else System.out.print(ch);
            }
            System.out.println();
        }
    }
}