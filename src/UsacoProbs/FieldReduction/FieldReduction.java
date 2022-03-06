package UsacoProbs.FieldReduction;

import java.io.*;
import java.util.*;

public class FieldReduction {

    static int minX, maxX;
    static int maxY, minY;
    static int minX2, maxX2;
    static int maxY2, minY2;

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("reduce.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("reduce.out")));

        int nCows = r.nextInt();
        ArrayList<Point> points = new ArrayList<>();
        for (int i = 0; i < nCows; i++) {
            int start = r.nextInt();
            int end = r.nextInt();
            points.add(new Point(start, end));
        }

        func(points);
        ArrayList<Point> left = new ArrayList<>();
        ArrayList<Point> right = new ArrayList<>();
        ArrayList<Point> up = new ArrayList<>();
        ArrayList<Point> down = new ArrayList<>();

        for (Point point : points) {
            if (point.x == minX)
                left.add(new Point(point.x, point.y));

            if (point.x == maxX)
                right.add(new Point(point.x, point.y));

            if (point.y == minY)
                down.add(new Point(point.x, point.y));

            if (point.y == maxY)
                up.add(new Point(point.x, point.y));
        }

        TreeSet<Integer> treeSet = new TreeSet<>();

        int area;
        if (left.size() == 1) {
            points.remove(left.get(0));
            func(points);
            treeSet.add((maxX - minX) * (maxY - minY));
            points.add(left.get(0));
        }

        if (right.size() == 1) {
            points.remove(right.get(0));
            func(points);
            treeSet.add((maxX - minX) * (maxY - minY));
            points.add(right.get(0));
        }

        if (up.size() == 1) {
            points.remove(up.get(0));
            func(points);
            treeSet.add((maxX - minX) * (maxY - minY));
            points.add(up.get(0));
        }

        if (down.size() == 1) {
            points.remove(down.get(0));
            func(points);
            treeSet.add((maxX - minX) * (maxY - minY));
            points.add(down.get(0));
        }

        treeSet.add((maxX - minX) * (maxY - minY));
        pw.println(treeSet.first());
        pw.close();
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

        @Override
        public String toString() {
            return "(" + x + ", " + y + ')';
        }
    }

    public static void func(ArrayList<Point> points) {
        points.sort(Comparator.comparingInt(o -> o.x));
        maxX = points.get(points.size() - 1).x;
        minX = points.get(0).x;

        maxX2 = points.get(points.size() - 2).x;
        minX2 = points.get(1).x;

        // y
        points.sort(Comparator.comparingInt(o -> o.y));
        maxY = points.get(points.size() - 1).y;
        minY = points.get(0).y;

        maxY2 = points.get(points.size() - 2).y;
        minY2 = points.get(1).y;
    }
}
