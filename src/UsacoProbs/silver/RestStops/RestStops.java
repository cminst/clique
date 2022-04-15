package UsacoProbs.silver.RestStops;

import java.io.*;
import java.util.Scanner;

public class RestStops {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("reststops.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("reststops.out")));

        r.nextInt();
        var nRestStops = r.nextInt();
        var johnSpeed = r.nextInt();
        var bessieSpeed = r.nextInt();

        var maxs = new boolean[nRestStops];

        var restStops = new int[nRestStops];
        var tastiness = new int[nRestStops];
        for (int i = 0; i < nRestStops; i++) {
            restStops[i] = r.nextInt();
            tastiness[i] = r.nextInt();
        }

        var max = 0;
        for (int i = nRestStops-1; i >= 0; i--) {
            if (tastiness[i] > max) {
                maxs[i] = true;
                max = tastiness[i];
            }
        }

        var maxTastinessUnits = 0L;
        var johnTheFarmer = 0L;
        var bessieTheCow = 0L;
        var lastRestStop = 0;
        for (int i = 0; i < nRestStops; i++) {
            if (maxs[i]) {
                johnTheFarmer += (long) (restStops[i] - lastRestStop) *johnSpeed;
                bessieTheCow += (long) (restStops[i] - lastRestStop) *bessieSpeed;
                maxTastinessUnits += (johnTheFarmer - bessieTheCow)*(tastiness[i]);
                bessieTheCow = johnTheFarmer;
                lastRestStop = restStops[i];
            }
        }
        pw.println(maxTastinessUnits);
        pw.close();
    }
}