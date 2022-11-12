package CsesProbs.ConcertTickets;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;

public class ConcertTickets {
    static InputReader r = new InputReader(System.in);
    static PrintWriter pw = new PrintWriter(System.out);

    public static int binarySearch(ArrayList<Integer> a, int key) {
        int low = 0;
        var high = a.size() - 1;
        while (low < high) {
            int mid = (low + high) / 2;
            int midVal = a.get(mid);
            if (midVal < key) {
                low = mid + 1;
                int lowVal = a.get(low);
                if (low == a.size() - 1 && lowVal <= key) return lowVal;
            } else if (midVal > key) {
                high = mid - 1;
                if (high == -1) {
                    return -1;
                }
                int highVal = a.get(high);
                if (high == 0 && highVal <= key) return highVal;
            } else {
                return midVal;
            }
        }
        if (low == high && a.get(low) <= key) return a.get(low);
        if (low == 0) return -1;
        return a.get(low - 1);
    }

    public static void main(String[] args) {
        int nTickets = r.nextInt();
        int mCustomers = r.nextInt();
        ArrayList<Integer> ticketPrice = new ArrayList<>();
        for (int i = 0; i < nTickets; i++) {
            ticketPrice.add(r.nextInt());
        }

        Collections.sort(ticketPrice);

        int i = 0;
        while (i < mCustomers) {
            int result = binarySearch(ticketPrice, r.nextInt());
            pw.println(result);
            if (result != -1) {
                ticketPrice.remove((Object) result);
            }

            i++;
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

        public int nextInt() { // reads in the next int
            return Integer.parseInt(next());
        }

    }
}
