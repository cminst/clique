package CodeForces.WhiteSheet;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class WhiteSheet {

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

    static InputReader r = new InputReader(System.in);

    public static void main(String[] args) {


        int lowerXWhite = r.nextInt();
        int lowerYWhite = r.nextInt();
        int higherXWhite = r.nextInt();
        int higherYWhite = r.nextInt();

        int lowerX = r.nextInt();
        int lowerY = r.nextInt();
        int higherX = r.nextInt();
        int higherY = r.nextInt();

        int lowerX2 = r.nextInt();
        int lowerY2 = r.nextInt();
        int higherX2 = r.nextInt();
        int higherY2 = r.nextInt();

        int length = Math.max(Math.max(Math.max(higherXWhite, higherYWhite), Math.max(higherX, higherY)), Math.max(higherX2, higherY2));
        int[][] billboards = new int[length][length];

        for (int i = lowerXWhite; i < higherXWhite; i++) {
            for (int j = lowerYWhite; j < higherYWhite; j++) {
                billboards[i][j] = 1;
            }
        }

        for (int i = lowerX; i < higherX; i++) {
            for (int j = lowerY; j < higherY; j++) {
                billboards[i][j] = 0;
            }
        }
        for (int i = lowerX2; i < higherX2; i++) {
            for (int j = lowerY2; j < higherY2; j++) {
                billboards[i][j] = 0;
            }
        }

        for (int i = 0; i < billboards.length/2; i++) {
            for (int j = 0; j < billboards.length/2; j++) {
                if (billboards[i][j] == 1) {
                    System.out.println("YES");
                    System.exit(0);
                }

                if (billboards[billboards.length-i][billboards.length-j] == 1) {
                    System.out.println("YES");
                    System.exit(0);
                }
            }
        }

        System.out.println("NO");
    }
}
