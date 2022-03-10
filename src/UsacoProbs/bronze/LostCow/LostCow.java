package UsacoProbs.bronze.LostCow;

import java.io.*;
import java.util.Scanner;

public class LostCow {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("lostcow.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("lostcow.out")));
        int farmerPosition = r.nextInt();
        int bessiePosition = r.nextInt();
        int xDirection = 1;
        int distanceFromX = 1;
        int totalDistance = 0;
        while (true) {
            if ((xDirection == 1 && farmerPosition <= bessiePosition && bessiePosition <= farmerPosition + distanceFromX) || (xDirection == -1 && bessiePosition <= farmerPosition && bessiePosition >= farmerPosition - distanceFromX)) {
                totalDistance += Math.abs(farmerPosition - bessiePosition);
                break;
            } else {
                totalDistance += distanceFromX * 2;
                distanceFromX *= 2;
                xDirection *= -1;
            }
        }

        pw.println(totalDistance);
        pw.close();
    }
}
