
import java.io.*;
import java.util.Scanner;

public class MountainView {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("mountains.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("mountains.out")));

        var nMountains = r.nextInt();
        var mountainPeaks = new Long[nMountains][6];
        for (int i = 0; i < nMountains; i++) {
            var x = 0L; var y = 0L;
            x = r.nextInt();
            y = r.nextInt();
            mountainPeaks[i][0] = x;
            mountainPeaks[i][1] = y;
            mountainPeaks[i][2] = x - y;
            mountainPeaks[i][3] = y - x;
            mountainPeaks[i][4] = x + y;
            mountainPeaks[i][5] = 0L;
        }
        var mountainsSeen = nMountains;
        for (int first = 0; first < nMountains; first++) {
            for (int second = 0; second < nMountains; second++) {
                if (first != second) {
                    if (mountainPeaks[second][4]-mountainPeaks[first][0] >= mountainPeaks[first][1] && mountainPeaks[first][0]-mountainPeaks[second][2]>= mountainPeaks[first][1]) {
                        mountainsSeen--;
                        break;
                    }
                }
            }
        }
        pw.println(mountainsSeen);
        pw.close();
    }
}
