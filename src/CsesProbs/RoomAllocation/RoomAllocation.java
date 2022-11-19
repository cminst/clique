package CsesProbs.RoomAllocation;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;

public class RoomAllocation {

    static InputReader r = new InputReader(System.in);
    static PrintWriter pw = new PrintWriter(System.out);

    public static void main(String[] args) {
        int customers = r.nextInt();
        ArrayList<StartEndPair> pairs = new ArrayList<>();
        for (int i = 0; i < customers; i++) {
            StartEndPair arrival = new StartEndPair(r.nextInt(), "s");
            StartEndPair departure = new StartEndPair(r.nextInt() + 1, "e");
            pairs.add(arrival);
            pairs.add(departure);
        }
        Collections.sort(pairs);
        int currentRooms = 0;
        int minRooms = 0;
        for (StartEndPair pair : pairs) {
            if (pair.se.equals("s")) {
                currentRooms++;
                if (currentRooms > minRooms) {
                    minRooms = currentRooms;
                }
            } else currentRooms--;
        }

        pw.println(minRooms);
        pw.close();
    }

    static class StartEndPair implements Comparable<StartEndPair> {
        public int time;
        public String se;


        public StartEndPair(int time, String se) {
            this.time = time;
            this.se = se;
        }

        public int compareTo(StartEndPair o) {
            if (this.time == o.time) {
                return this.se.compareTo(o.se);
            }
            return Integer.compare(this.time, o.time);
        }
    }

    static class InputReader {
        BufferedReader reader;
        StringTokenizer tokenizer;

        public InputReader(InputStream stream) {
            reader = new BufferedReader(new InputStreamReader(stream), 32768);
            tokenizer = null;
        }

        String next() { // reads in the next string
            while (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }

        public int nextInt() { // reads in the next int
            return Integer.parseInt(next());
        }
    }
}
