package UsacoProbs.silver.FieldReduction;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class FieldReduction2 {
    static int n;

    static class Analysis {
        long area;
        ArrayList<ArrayList<Integer>> borders;
    }

    public static Analysis analyze(ArrayList<Integer> indices, int[] xs, int[] ys) {
        var minX = 1000000000;
        var minY = 1000000000;
        var maxX = -1000000000;
        var maxY = -1000000000;
        for (int i = 0; i < n; i++) {
            var skip = false;
            for (Integer index : indices) {
                if (index == i) {
                    skip = true;
                }
            }
            if (skip) {
                continue;
            }

            minX = Math.min(minX, xs[i]);
            maxX = Math.max(maxX, xs[i]);
            minY = Math.min(minY, ys[i]);
            maxY = Math.max(maxY, ys[i]);
        }
        Analysis analysis = new Analysis();
        analysis.area = (long) (maxX - minX) * (maxY - minY);

        var left = new ArrayList<Integer>();
        var right = new ArrayList<Integer>();
        var up = new ArrayList<Integer>();
        var down = new ArrayList<Integer>();

        for (int i = 0; i < n; i++) {
            var skip = false;
            for (Integer index : indices) {
                if (index == i) {
                    skip = true;
                }
            }

            if (skip) continue;

            if (xs[i] == minX) left.add(i);
            if (xs[i] == maxX) right.add(i);
            if (ys[i] == minY) up.add(i);
            if (ys[i] == maxY) down.add(i);
        }
        analysis.borders = new ArrayList<>();
        if (up.size() <= 3) analysis.borders.add(up);
        if (down.size() <= 3) analysis.borders.add(down);
        if (left.size() <= 3) analysis.borders.add(left);
        if (right.size() <= 3) analysis.borders.add(right);

        return analysis;
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("reduce2.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("reduce2.out")));

        n = r.nextInt();
        var xs = new int[n];
        var ys = new int[n];
        for (int i = 0; i < n; i++) {
            xs[i] = r.nextInt();
            ys[i] = r.nextInt();
        }

        var list = analyze(new ArrayList<>(), xs, ys);
        var minArea = list.area;
        for (ArrayList<Integer> points : list.borders) {
            var smallerAnalysis = analyze(points, xs, ys);
            minArea = Math.min(minArea, smallerAnalysis.area);
            for (ArrayList<Integer> points2 : smallerAnalysis.borders) {
                if (points2.size() + points.size() <= 3) {
                    points2.addAll(points);
                    var evenSmallerAnalysis = analyze(points2, xs, ys);
                    minArea = Math.min(minArea, evenSmallerAnalysis.area);
                    for (ArrayList<Integer> points3 : evenSmallerAnalysis.borders) {
                        if (points2.size() + points3.size() <= 3) {
                            points3.addAll(points2);
                            var superTinyAnalysis = analyze(points3, xs, ys);
                            minArea = Math.min(minArea, superTinyAnalysis.area);
                        }
                    }
                }
            }
        }
        pw.println(minArea);
        pw.close();
    }
}