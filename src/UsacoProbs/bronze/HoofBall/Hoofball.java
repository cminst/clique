package UsacoProbs.bronze.HoofBall;

import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public class Hoofball {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("hoofball.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("hoofball.out")));

        int nCows = r.nextInt();
        int[] cows = new int[nCows];
        for (int i = 0; i < nCows; i++) cows[i] = r.nextInt();
        Arrays.sort(cows);

        int[] distances = new int[nCows - 1];
        for (int i = 0; i < nCows - 1; i++) distances[i] = cows[i + 1] - cows[i];

        var balls = 0;

        var direction = Direction.DEVAULT;
        for (int i = 0; i < distances.length - 1; i++) {
            if (distances[i] > distances[i + 1] && direction == Direction.UP) {
                direction = Direction.DEVAULT;
                balls++;
                continue;
            } else if (distances[i] <= distances[i + 1] && direction == Direction.DOWN) {
                balls++;
                if (i < distances.length - 2) {
                    if (distances[i + 1] > distances[i + 2]) i++;
                }
                direction = Direction.DEVAULT;
                continue;
            }

            if (distances[i] <= distances[i + 1]) direction = Direction.UP;
            else direction = Direction.DOWN;
        }
        pw.println(balls + 1);
        pw.close();
    }

    enum Direction {UP, DOWN, DEVAULT}
}
