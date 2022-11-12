package CsesProbs.ChessboardQueens;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Objects;
import java.util.StringTokenizer;

public class ChessboardQueens {

    static InputReader r = new InputReader(System.in);
    static int count = 0;
    static ArrayList<Point> reserved = new ArrayList<>();

    public static void main(String[] args) {
        for (int i = 0; i < 8; i++) {
            String str = r.next();
            for (int j = 0; j < 8; j++) {
                if (str.charAt(j) == '*') {
                    reserved.add(new Point(i, j));
                }
            }
        }
        getQueens(reserved, 0);
        System.out.println(count);
    }

    private static void getQueens(ArrayList<Point> noSquares, int row) {
        if (row < 8) {
            for (int i = 0; i < 8; i++) {
                if (!noSquares.contains(new Point(row, i))) {
                    noSquares.add(new Point(row, i));
                    for (int j = 1; j < 8; j++) {
                        noSquares.add(new Point(row, i - j));
                        noSquares.add(new Point(row, i + j));
                        noSquares.add(new Point(row - j, i));
                        noSquares.add(new Point(row + j, i));
                        // diagonal
                        noSquares.add(new Point(row + j, i + j));
                        noSquares.add(new Point(row - j, i - j));
                        noSquares.add(new Point(row - j, i + j));
                        noSquares.add(new Point(row + j, i - j));
                    }
                    getQueens(noSquares, row + 1);
                    for (int j = 0; j < 57; j++) {      // for running
                        noSquares.remove(noSquares.size() - 1);
                    }
                }
            }
        } else {
            count++;
        }
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

        public long nextLong() { // reads in the next long
            return Long.parseLong(next());
        }

        public double nextDouble() { // reads in the next double
            return Double.parseDouble(next());
        }
    }

    static class Point {
        public int x;
        public int y;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }

        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Point point = (Point) o;
            return x == point.x && y == point.y;
        }

        public int hashCode() {
            return Objects.hash(x, y);
        }
    }
}