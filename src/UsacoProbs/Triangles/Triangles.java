package UsacoProbs.Triangles;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

public class Triangles {

    static void addOccurrences(int x, int y, HashMap<Integer, HashSet<Integer>> points) {
        if (points.containsKey(x)) {
            points.get(x).add(y);
        } else {
            HashSet<Integer> hashset = new HashSet<>();
            hashset.add(y);
            points.put(x, hashset);
        }
    }

    static void check(int[] arr) {

    }

    static void swap(int[] arr, int index1, int index2) {
        int first = arr[index1];
        int second = arr[index2];
        arr[index1] = second;
        arr[index2] = first;
    }

    static void generate(int[] arr, int k) {
        if (k == 1) {
            check(arr);
        }
        else {
            generate(arr, k-1);
            for (int i = 0; i < k - 1; i++) {
                if (k % 2 == 0) {
                    swap(arr, i, k-1);
                }
                else {
                    swap(arr, 0, k-1);
                }
                generate(arr, k-1);
            }
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
            addOccurrences(x, y, points);
        }
        int previousY = 0;
        int maxArea = 0;
        for (Integer x : points.keySet()) {
            HashSet<Integer> xs = new HashSet<>();
            HashSet<Integer> ys = points.get(x);
            for (Integer y : ys) {
                if (previousY != y) {
                    for (Integer i : points.keySet()) {
                        if (points.get(i).contains(y)) {
                            xs.add(i);
                        }
                    }
//                    pw.println(xs);
                    int maxY = 0;
                    int minY = Integer.MAX_VALUE;
                    for (int i : ys) {
                        if (i > maxY) {
                            maxY = i;
                        }
                        if (i < minY) {
                            minY = i;
                        }
                    }
                    int maxX = 0;
                    for (int i : xs) {
                        if (i > maxX) {
                            maxX = i;
                        }
                    }
                    if (ys.size() > 1) {
                        if ((maxY - minY) * (maxX - minY) > maxArea) {
                            maxArea = (maxY - minY) * (maxX - minY);
                        }
                    }
                    previousY = y;
                }
            }
        }
        pw.println(maxArea);
        pw.close();
    }
}