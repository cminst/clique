package UsacoProbs.MilkMeasurement;

import java.io.*;
import java.util.*;

public class MilkMeasurement {

    static class Log implements Comparable<Log> {
        public int log;
        public String name;
        public int changedOutput;

        public Log(int log, String name, int changedOutput) {
            this.log = log;
            this.name = name;
            this.changedOutput = changedOutput;
        }

        public int compareTo(Log o) {
            return Integer.compare(this.log, o.log);
        }
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("measurement.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("measurement.out")));
        int numLogs = r.nextInt();
        Log[] logs = new Log[numLogs];
        for (int i = 0; i < numLogs; i++) {
            logs[i] = new Log(r.nextInt(), r.next(), r.nextInt());
        }
        Arrays.sort(logs);
        int BessieMilk = 7;
        int ElsieMilk = 7;
        int MildredMilk = 7;

        boolean[] pastLeaderboard = new boolean[]{false, false, false};
        int count = 0;

        for (Log log : logs) {
            if (log.name.equals("Bessie")) {
                BessieMilk += log.changedOutput;
            }
            else if (log.name.equals("Elsie")) {
                ElsieMilk += log.changedOutput;
            }
            else {
                MildredMilk += log.changedOutput;
            }
            boolean[] leaderboard = new boolean[]{false, false, false};
            int belsiemilk = Math.max(BessieMilk, ElsieMilk);
            int maxMilk = Math.max(belsiemilk, MildredMilk);
            if (maxMilk == BessieMilk) leaderboard[0] = true;
            if (maxMilk == ElsieMilk) leaderboard[1] = true;
            if (maxMilk == MildredMilk) leaderboard[2] = true;
            if (!Arrays.equals(leaderboard, pastLeaderboard)) {
                pastLeaderboard = leaderboard;
                count++;
            }
        }
        pw.println(count);
        pw.close();
    }
}
