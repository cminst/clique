package CodeForces.SortedAdjacentDifferences;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class SortedAdjacentDifferences {

    public static void main(String[] args) {
        Scanner r = new Scanner(System.in);
        var tTestcases = r.nextInt();
        for (int t = 0; t < tTestcases; t++) {
            var n = r.nextInt();
            var result = new ArrayList<Integer>();
            var integers = new ArrayList<Integer>();
            for (int j = 0; j < n; j++) {
                integers.add(r.nextInt());
            }
            Collections.sort(integers);
            var index = 0;
            var count = integers.size() - 1;
            while (count != 0) {
                result.add(0, integers.get(index));
                index += count;
                if (count > 0) count = count - 1;
                else count = count + 1;
                count *= -1;
            }
            result.add(0, integers.get(index));
            for (Integer integer : result) {
                System.out.println(integer);
            }
        }
    }
}
