
import java.io.*;
import java.util.Scanner;

public class PromotionCounting {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("promote.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("promote.out")));

        var before = new int[4];
        var after = new int[4];
        for (int i = 0; i < 4; i++) {
            before[i] = r.nextInt();
            after[i] = r.nextInt();
        }
        var newParticipants = (after[0]+after[1]+after[2]+after[3]) - (before[0]+before[1]+before[2]+before[3]);

        before[0] += newParticipants;
        var bronzePromoted = before[0]-after[0];
        before[0] -= bronzePromoted;
        before[1] += bronzePromoted;
        var silverPromoted = before[1]-after[1];
        before[1] -= silverPromoted;
        before[2] += silverPromoted;
        var goldPromoted = before[2]-after[2];
        before[2] -= goldPromoted;

        pw.println(bronzePromoted);
        pw.println(silverPromoted);
        pw.println(goldPromoted);
        pw.close();
    }
}
