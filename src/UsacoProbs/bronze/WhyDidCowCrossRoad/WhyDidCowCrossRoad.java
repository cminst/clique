package UsacoProbs.bronze.WhyDidCowCrossRoad;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class WhyDidCowCrossRoad {

    static class StartEndPair implements Comparable<StartEndPair> {
        public int start;
        public int questionTime;

        public StartEndPair(int start, int questionTime) {
            this.start = start;
            this.questionTime = questionTime;
        }

        public int compareTo(StartEndPair o) {
            if (this.start == o.start) {
                return Integer.compare(this.questionTime, o.questionTime);
            }
            return Integer.compare(this.start, o.start);
        }
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("cowqueue.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("cowqueue.out")));
        int numCows = r.nextInt();
        ArrayList<StartEndPair> pairs = new ArrayList<>();
        for (int i = 0; i < numCows; i++) {
            int start = r.nextInt();
            int questionTime = r.nextInt();
            StartEndPair pair = new StartEndPair(start, questionTime);
            pairs.add(pair);
        }
        Collections.sort(pairs);
        int time = pairs.get(0).start;
        for (int i = 0; i < numCows; i++) {
            if (pairs.get(i).start > time) time += pairs.get(i).start - time;
            time += pairs.get(i).questionTime;
        }
        pw.println(time);
        pw.close();
    }
}