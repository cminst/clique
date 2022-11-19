package UsacoProbs.silver.MooTube;

import java.io.*;
import java.util.LinkedList;
import java.util.Scanner;

public class MooTube {
    static LinkedList<Edge>[] pairOfVids = null;

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("mootube.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("mootube.out")));
        int nVids = r.nextInt();
        int qQuestions = r.nextInt();

        pairOfVids = new LinkedList[nVids];
        for (int i = 0; i < nVids; i++) pairOfVids[i] = new LinkedList<>();
        for (int i = 1; i < nVids; i++) {
            var firstVid = r.nextInt() - 1;
            var secondVid = r.nextInt() - 1;
            var w = r.nextInt();
            pairOfVids[firstVid].add(new Edge(secondVid, w));
            pairOfVids[secondVid].add(new Edge(firstVid, w));
        }
        for (int i = 0; i < qQuestions; i++) {
            var minK = r.nextInt();
            var vidNum = r.nextInt() - 1;
            var count = 0;
            var vids = new LinkedList<Integer>();
            vids.add(vidNum);
            var seenVids = new boolean[nVids];
            seenVids[vidNum] = true;
            while (!vids.isEmpty()) {
                var vid = vids.removeFirst();
                for (Edge j : pairOfVids[vid]) {
                    var jVid = j.vid;
                    var weight = j.w;
                    if (weight >= minK && !seenVids[jVid]) {
                        seenVids[jVid] = true;
                        vids.add(jVid);
                        count++;
                    }
                }
            }
            pw.println(count);
        }
        pw.close();
    }

    static class Edge {
        int vid, w;

        public Edge(int vid, int w) {
            this.vid = vid;
            this.w = w;
        }
    }
}