package UsacoProbs.BackAndForth;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

public class BackAndForth {

    public static void addOccurrences(HashMap<Integer, Integer> hashMap, int bucketSize) {
        if (!hashMap.containsKey(bucketSize)) {
            hashMap.put(bucketSize, 0);
        }
        hashMap.put(bucketSize, hashMap.get(bucketSize) + 1);
    }
    public static void removeOccurrences(HashMap<Integer, Integer> hashMap, int bucketSize) {
        if (hashMap.get(bucketSize) == 1) {
            hashMap.remove(bucketSize);
        }
        else {
            hashMap.put(bucketSize, hashMap.get(bucketSize) - 1);
        }
    }

    static HashSet<Integer> possMilk = new HashSet<>();
    static HashMap<Integer, Integer> differentBucketSizes1 = new HashMap<>();
    static HashMap<Integer, Integer> differentBucketSizes2 = new HashMap<>();

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("backforth.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("backforth.out")));
        for (int i = 0; i < 10; i++) {
            addOccurrences(differentBucketSizes1, r.nextInt());
        }
        for (int i = 0; i < 10; i++) {
            addOccurrences(differentBucketSizes2, r.nextInt());
        }
        possibleResults(1, 0);
        pw.println(possMilk.size());
        pw.close();
    }

    static ArrayList<Integer> milk = new ArrayList<>();

    private static boolean possibleResults(int count, int milk) {
        HashMap<Integer, Integer> bucketSizes1 = differentBucketSizes1;
        HashMap<Integer, Integer> bucketSizes2 = differentBucketSizes2;
        int n = -1;
        if(count % 2 == 0) {
            bucketSizes1 = differentBucketSizes2;
            bucketSizes2 = differentBucketSizes1;
            n = 1;
        }

        for (int i : ((HashMap<Integer, Integer>) bucketSizes1.clone()).keySet()) {
            if (count != 4) {
                removeOccurrences(bucketSizes1, i);
                addOccurrences(bucketSizes2, i);
                possibleResults(count+1, milk+i*n);
                addOccurrences(bucketSizes1, i);
                removeOccurrences(bucketSizes2, i);
            }
            else possMilk.add(milk+i*n);
        }
        return false;
    }
}
