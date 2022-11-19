package CsesProbs.TrafficLights;

import java.io.*;
import java.util.Collections;
import java.util.PriorityQueue;
import java.util.StringTokenizer;
import java.util.TreeSet;

public class TrafficLights {
    static InputReader r = new InputReader(System.in);
    static PrintWriter pw = new PrintWriter(System.out);
    static PriorityQueue<Integer> passageSizes = new PriorityQueue<>(Collections.reverseOrder());
    static TreeSet<Integer> trafficLights = new TreeSet<>();

    public static void findClosest(int trafficLight) {
        Integer high = trafficLights.higher(trafficLight);
        Integer low = trafficLights.lower(trafficLight);
        if (high == null || low == null) {
            return;
        }

        passageSizes.remove(high - low);
        passageSizes.add(high - trafficLight);
        passageSizes.add(trafficLight - low);
    }

    public static void main(String[] args) {
        int streetLen = r.nextInt();
        int numTrafficLights = r.nextInt();
        trafficLights.add(0);
        trafficLights.add(streetLen);
        for (int i = 0; i < numTrafficLights; i++) {
            int trafficLight = r.nextInt();
            findClosest(trafficLight);
            trafficLights.add(trafficLight);
            pw.println(passageSizes.peek() + " ");
        }
        pw.close();
    }

    static class InputReader {
        BufferedReader reader;
        StringTokenizer tokenizer;

        public InputReader(InputStream stream) {
            reader = new BufferedReader(new InputStreamReader(stream), 32768);
            tokenizer = null;
        }

        String next() {
            while (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }

        public int nextInt() {
            return Integer.parseInt(next());
        }
    }
}
