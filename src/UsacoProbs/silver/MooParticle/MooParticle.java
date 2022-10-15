package UsacoProbs.silver.MooParticle;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;

public class MooParticle {

    static class Particle implements Comparable<Particle> {
        public int x,y;
        public Particle(int x, int y) {
            this.x = x;
            this.y = y;
        }

        public int compareTo(Particle particle) {
            if(x==particle.x) return Integer.compare(y, particle.y);
            return Integer.compare(x,particle.x);
        }
    }

    public static void main(String[] args) throws IOException {
        BufferedReader r = new BufferedReader(new FileReader("moop.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("moop.out")));

        var n = Integer.parseInt(r.readLine());
        var sorted = new ArrayList<Particle>();
        for (int i = 0; i < n; i++) {
            var st = new StringTokenizer(r.readLine());
            var x = Integer.parseInt(st.nextToken());
            var y = Integer.parseInt(st.nextToken());
            sorted.add(new Particle(x,y));
        }
        Collections.sort(sorted);

        var minL = new int[n];
        minL[0] = sorted.get(0).y;
        for (int i = 1; i < n; i++) minL[i] = Math.min(minL[i-1], sorted.get(i).y);
        var maxR = new int[n];
        maxR[n-1] = sorted.get(n-1).y;
        for (int i = n-2; i >=0; i--) maxR[i] = Math.max(maxR[i+1], sorted.get(i).y);
        var count = 1;
        for (int i = 0; i < n-1; i++) {
            if(minL[i]>maxR[i+1]) count++;
        }
        pw.println(count);
        pw.close();
    }
}
