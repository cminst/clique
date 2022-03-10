package UsacoProbs.bronze.CircularBarn;

import java.io.*;
import java.util.Scanner;

public class CircularBarn {

    static int getCount(int position, int[] roomCows) {
        return roomCows[position % roomCows.length];
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("cbarn.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("cbarn.out")));
        int nRooms = r.nextInt();
        int[] roomCows = new int[nRooms];
        for (int i = 0; i < nRooms; i++) {
            roomCows[i] = r.nextInt();
        }
        int maxDistance = Integer.MAX_VALUE;
        for (int i = 0; i < nRooms; i++) {
            int cowDistance = 0;
            for (int j = 0; j < nRooms; j++) {
                cowDistance += j * getCount(i + j, roomCows);
            }
            if (cowDistance < maxDistance) {
                maxDistance = cowDistance;
            }
        }
        pw.println(maxDistance);
        pw.close();
    }
}
