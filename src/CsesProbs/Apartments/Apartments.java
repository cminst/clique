package CsesProbs.Apartments;

import java.io.*;
import java.util.Arrays;
import java.util.StringTokenizer;

public class Apartments {
    static InputReader r = new InputReader(System.in);
    static PrintWriter pw = new PrintWriter(System.out);

    public static void main(String[] args) {
        int numApplicantApartment = r.nextInt();
        int numApartment = r.nextInt();
        int maxk = r.nextInt();
        int[] applicantApartmentSizes = new int[numApplicantApartment];
        for (int i = 0; i < numApplicantApartment; i++) {
            applicantApartmentSizes[i] = r.nextInt();
        }
        int[] apartmentSizes = new int[numApartment];
        for (int i = 0; i < numApartment; i++) {
            apartmentSizes[i] = r.nextInt();
        }
        Arrays.sort(applicantApartmentSizes);
        Arrays.sort(apartmentSizes);

        int result = 0, i = 0, j = 0;
        while (i < numApplicantApartment) {
            if (j == numApartment) {
                break;
            }
            if (applicantApartmentSizes[i] > apartmentSizes[j] + maxk) {
                j++;
                continue;
            } else if (applicantApartmentSizes[i] < apartmentSizes[j] - maxk) {
                i++;
                continue;
            }
            if (apartmentSizes[j] >= applicantApartmentSizes[i] - maxk && apartmentSizes[j] <= applicantApartmentSizes[i] + maxk) {
                result++;
                i++;
                j++;
            }
        }
        pw.println(result);
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