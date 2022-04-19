package UsacoProbs.silver.WhyCowCrossedRoad;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class WhyCowCrossedRoad {

    static class Event implements Comparable<Event> {
        int start;
        int end;

        public Event(int s, int e) {
            start = s;
            end = e;
        }

        public int compareTo(Event e) {
            return Integer.compare(this.end, e.end);
        }
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("helpcross.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("helpcross.out")));

        var cChickens = r.nextInt();
        var nCows = r.nextInt();
        var chickenTimes = new ArrayList<Integer>();
        for (int i = 0; i < cChickens; i++) {
            chickenTimes.add(r.nextInt());
        }

        var cowTimes = new Event[nCows];
        for (int i = 0; i < nCows; i++) {
            var start = r.nextInt();
            var end = r.nextInt();
            cowTimes[i] = new Event(start, end);
        }

        Arrays.sort(cowTimes);

        var maxPairs = 0;
        Collections.sort(chickenTimes);
        for (int i = 0; i < nCows; i++) {
            var chicken = -1;
            for (Integer chickenTime : chickenTimes) {
                if (chickenTime >= cowTimes[i].start) {
                    chicken = chickenTime;
                    break;
                }
            }
            if (chicken!=-1) {
                if (chicken <= cowTimes[i].end) {
                    maxPairs++;
                    chickenTimes.remove((Integer) chicken);
                }
            }
        }
        pw.println(maxPairs);
        pw.close();
    }
}