package UsacoProbs.bronze.Teleportation;

import java.io.*;
import java.util.Scanner;

public class Teleportation {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("teleport.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("teleport.out")));

        int start = r.nextInt();
        int barn = r.nextInt();
        int teleportX = r.nextInt();
        int teleportY = r.nextInt();
        var min = Math.min(barn - start, Math.abs(start - teleportX) + Math.abs(barn - teleportY));
        min = Math.min(Math.abs(min), Math.abs(start - teleportY) + Math.abs(barn - teleportX));
        pw.println(Math.abs(min));
        pw.close();
    }
}
