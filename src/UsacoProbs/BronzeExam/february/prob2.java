package UsacoProbs.BronzeExam.february;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Scanner;

public class prob2 {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(System.in);

        int nCows = r.nextInt();

        var a = new ArrayList<Integer>();
        var b = new ArrayList<Integer>();
        var same = true;

        for (int i = 0; i < nCows; i++) {
            a.add(r.nextInt());
        }
        for (int i = 0; i < nCows; i++) {
            b.add(r.nextInt());
            if (!Objects.equals(b.get(i), a.get(i))) {
                same = false;
            }
        }

//        System.out.println(a);
//        var cow = a.get(a.size()-1);
//        System.out.println(cow);
//        a.remove(cow);
//        a.add(0, cow);
//        System.out.println(a);

        if (same) {
            System.out.println(0);
        } else {
            var changes = 0;
            while (!a.equals(b)) {
                var farthest = b.indexOf(a.get(0));
                var cowIndex = 0;
                for (int i = 0; i < a.size(); i++) {
                    if (Math.abs(i - b.indexOf(a.get(i))) > Math.abs(farthest)) {
                        farthest = b.indexOf(a.get(i)) - i;
                        cowIndex = i;
                    }
                }
                a.add(b.indexOf(a.get(cowIndex)), a.get(cowIndex));
                if (farthest < 0) {
                    a.remove(cowIndex+1);
                }
                else {
                    a.remove(cowIndex);
                }
                changes++;
            }
            System.out.println(changes);
        }
    }
}