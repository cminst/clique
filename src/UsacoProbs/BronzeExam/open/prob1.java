package UsacoProbs.BronzeExam.open;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class prob1 {
    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(System.in);

        var nCows = r.nextInt();
        var greaterThan = 0;
        var lessThan = Integer.MAX_VALUE;

        var gs = new ArrayList<Integer>();
        var ls = new ArrayList<Integer>();
        var Gcount = 0;
        var Lcount = 0;
        for (int i = 0; i < nCows; i++) {
            var lOrg = r.next();
            var num = r.nextInt();
            if (lOrg.equals("G")) {
                gs.add(num);
                if (num + 1 >= lessThan) {
                    var length = ls.size();
                    for (int j = 0; j < ls.size(); j++) {
                        if (ls.get(j) <= num + 1) {
                            ls.remove(j);
                        }
                    }
                    Lcount += length - ls.size();
                }
                if (num > greaterThan) {
                    greaterThan = num;
                }
            } else if (lOrg.equals("L")) {
                if (num - 1 <= greaterThan) {
                    ls.add(num);
                    var length = gs.size();
                    for (int j = 0; j < gs.size(); j++) {
                        if (gs.get(j) >= num - 1) {
                            gs.remove(j);
                        }
                    }
                    Gcount += length - gs.size();
                }
                if (num < lessThan) {
                    lessThan = num;
                }
            }
        }
        System.out.println(Math.min(Gcount, Lcount));
        //        System.out.println(Gcount);
    }
}
