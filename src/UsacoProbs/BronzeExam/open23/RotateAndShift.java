package UsacoProbs.BronzeExam.open23;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class RotateAndShift {
    public static void main(String[] args) throws IOException {
        BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
        var st = new StringTokenizer(r.readLine());

        var n = Integer.parseInt(st.nextToken());
        var k = Integer.parseInt(st.nextToken());
        var t = Integer.parseInt(st.nextToken());
        var a = new int[k+1];
        st = new StringTokenizer(r.readLine());
        for (int i = 0; i < k; i++) {
            a[i] = Integer.parseInt(st.nextToken());
        }
        a[k] = n;
        var cows = new int[n];
        for (int i = 0; i < k; i++) {
            for (int j = a[i]; j < a[i + 1]; j++) {
                var rotate = t-(j-a[i]+1);
                if (rotate>=0) {
                    var diff = a[i+1]-a[i];
                    var nextRotates = rotate/diff+1;
                    cows[(j+nextRotates*diff)%n] = j;
                }
                else {
                    cows[j] = j;
                }
            }
        }
        var str = new StringBuilder();
        for (int c : cows) {
            str.append(c).append(" ");
        }
        str.deleteCharAt(str.length()-1);
        System.out.println(str);
    }
}