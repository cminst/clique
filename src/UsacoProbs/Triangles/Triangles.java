package UsacoProbs.Triangles;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

public class Triangles {

    static void addOccurences(int x, int y, HashMap<Integer, HashSet<Integer>> points) {
        if (points.containsKey(x)) {
            points.get(x).add(y);
        } else {
            HashSet<Integer> hashset = new HashSet<>();
            hashset.add(y);
            points.put(x, hashset);
        }
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("triangles.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("triangles.out")));
        int nPoints = r.nextInt();
        HashMap<Integer, HashSet<Integer>> points = new HashMap<>();
        for (int i = 0; i < nPoints; i++) {
            int x = r.nextInt();
            int y = r.nextInt();
            addOccurences(x, y, points);
        }
        for (Integer x : points.keySet()) {
            HashSet<Integer> xs = new HashSet<>();
            HashSet<Integer> ys = points.get(x);
            for (Integer y : ys) {
                for (Integer i : points.keySet()) {
                    if (points.get(i).contains(y)) {
                        xs.add(i);
                    }
                }
                pw.println(xs);
                xs.clear();
            }
//            pw.print(x);
//            pw.println(ys);
        }
        pw.close();
    }
}
//        int[] xs = new int[nPoints];
//        int[] ys = new int[nPoints];
//        int[][] arr2d = new int[20000][20000];
//
//        for (int i = 0; i < arr2d.length; i++) {
//            for (int j = 0; j < arr2d.length; j++) {
//                arr2d[i][j] = -1;
//            }
//        }
//        for (int i = 0; i < nPoints; i++) {
//            int x = r.nextInt();
//            int y = r.nextInt();
//            xs[i] = x;
//            ys[i] = y;
//            arr2d[x][y] = 1;
//        }
//        int maxXDist = 0;
//        int maxYDist = 0;
//        for (int i = 0; i < nPoints; i++) {
//            for (int j = i + 1; j < nPoints; j++) {
//                int dx = Math.abs(xs[i] - xs[j]);
//                int dy = Math.abs(ys[i] - ys[j]);
//                if (dy + dx > maxXDist+ maxYDist) {
//                    maxXDist = dx;
//                    maxYDist = dy;
//                }
//            }
//        }
//
//        int maxXPoint = 0;
//        int maxYPoint = 0;
//        for (int i = 0; i < nPoints; i++) {
//            int x = xs[i];
//            int y = ys[i];
//            if (x+y > maxXPoint+maxYPoint) {
//                maxXPoint = x;
//                maxYPoint = y;
//            }
//        }
//
//        for (int i = 0; i < xs.length; i++) {
//            if (xs[i] == maxXPoint-maxXDist && ys[i] == maxYPoint-maxYDist) {
//                if ((xs[i] == maxXPoint && ys[i] == maxYPoint-maxYDist) || (xs[i] == maxXPoint-maxXDist && ys[i] == maxYPoint)) {
//                    pw.println(maxXDist*maxYDist);
//                }
//            }
//        }