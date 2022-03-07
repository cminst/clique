
import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class CowEvolution {
    static int nSubPopulations;
    static ArrayList<String> features = new ArrayList<>();
    static ArrayList<ArrayList<String>> eachFeatures = new ArrayList<>();

    static boolean crosses(int a, int b) {
        boolean A = false, B = false, AandB = false;

        for (int i = 0; i < nSubPopulations; i++) {
            ArrayList<String> iFeatures = eachFeatures.get(i);

            boolean hasA = false, hasB = false;
            if (iFeatures.contains(features.get(a))) hasA = true;
            if (iFeatures.contains(features.get(b))) hasB = true;

            if (hasA && hasB) AandB = true;
            else if (hasA) A = true;
            else if (hasB) B = true;

            if (AandB && A && B) return true;
        }
        return false;
    }

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(new FileReader("evolution.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("evolution.out")));
        nSubPopulations = r.nextInt();
        for (int i = 0; i < nSubPopulations; i++) {
            var kCharacteristics = r.nextInt();
            var characteristics = new ArrayList<String>();
            for (int j = 0; j < kCharacteristics; j++) {
                var str = r.next();

                if (!features.contains(str)) {
                    features.add(str);
                }
                characteristics.add(str);
            }
            eachFeatures.add(characteristics);
        }

        for (int j = 0; j < features.size()-1; j++) {
            for (int k = j + 1; k < features.size(); k++) {
                if (crosses(j, k)) {
                    pw.println("no");
                    pw.close();
                    return;
                }
            }
        }

        pw.println("yes");
        pw.close();
    }
}