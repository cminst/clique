package UsacoProbs.bronze.BucketBrigade;

import java.awt.*;
import java.io.*;
import java.util.Scanner;

public class BucketBrigade {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("buckets.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("buckets.out")));

        var barn = new Point();
        var rock = new Point();
        var lake = new Point();
        for (int i = 0; i < 10; i++) {
            var str = r.next();
            if (str.contains("B")) {
                barn.x = i;
                barn.y = str.indexOf("B");
            }
            if (str.contains("R")) {
                rock.x = i;
                rock.y = str.indexOf("R");
            }
            if (str.contains("L")) {
                lake.x = i;
                lake.y = str.indexOf("L");
            }
        }

        if (lake.x == barn.x && rock.x == lake.x && rock.y < Math.max(barn.y, lake.y) && rock.y > Math.min(barn.y, lake.y))
            pw.println(Math.abs(barn.y - lake.y) + 1);
        else if (lake.y == barn.y && rock.y == lake.y && rock.x < Math.max(barn.x, lake.x) && rock.x > Math.min(barn.x, lake.x))
            pw.println(Math.abs(barn.x - lake.x) + 1);
        else pw.println(Math.abs(barn.y - lake.y) + Math.abs(barn.x - lake.x) - 1);

        pw.close();
    }
}
