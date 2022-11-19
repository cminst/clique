package UsacoProbs.silver.Moocast;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class Moocast {

    static ArrayList<Cow> cows;

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("moocast.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("moocast.out")));

        var nCows = r.nextInt();
        cows = new ArrayList<>();
        for (int i = 0; i < nCows; i++) {
            var x = r.nextInt();
            var y = r.nextInt();
            var distance = r.nextInt();
            cows.add(new Cow(x, y, distance));
        }
        var maxCount = 0;
        for (Cow cow : cows) {
            maxCount = Math.max(maxCount, recursiveFun(cow, new ArrayList<>()));
        }
        pw.println(maxCount);
        pw.close();
    }

    static private int recursiveFun(Cow c, ArrayList<Cow> neighbors) {
        for (Cow cow : cows) {
            if (Math.sqrt(Math.pow(Math.abs(c.x - cow.x), 2.0) + Math.pow(Math.abs(c.y - cow.y), 2.0)) <= c.distance) {
                if (!neighbors.contains(cow)) {
                    neighbors.add(cow);
                    recursiveFun(cow, neighbors);
                }
            }
        }
        return neighbors.size();
    }

    static class Cow {
        int x, y, distance;

        public Cow(int x, int y, int distance) {
            this.x = x;
            this.y = y;
            this.distance = distance;
        }
    }
}
