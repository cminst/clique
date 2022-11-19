package UsacoProbs.silver.SubsequencesSevens;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;

public class SubsequencesSevens {

    public static void main(String[] args) throws IOException {
        BufferedReader r = new BufferedReader(new FileReader("div7.in"));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("div7.out")));

        var st = new StringTokenizer(r.readLine());
        var n = Integer.parseInt(st.nextToken());
        var prefix = new int[n];
        st = new StringTokenizer(r.readLine());
        prefix[0] = Integer.parseInt(st.nextToken())%7;
        for (int i = 1; i < n; i++) {
            st = new StringTokenizer(r.readLine());
            prefix[i] = (prefix[i-1]+Integer.parseInt(st.nextToken()))%7;
        }
        var hashMap = new HashMap<Integer, ArrayList<Integer>>();
        for (int i = 0; i < prefix.length; i++) {
            hashMap.putIfAbsent(prefix[i], new ArrayList<>());
            var arr = hashMap.get(prefix[i]);
            arr.add(i);
            hashMap.put(prefix[i], arr);
        }
        var max = 0;
        for (int key : hashMap.keySet()) {
            var list = hashMap.get(key);
            if(list.get(list.size()-1)-list.get(0)>max) {
                max = list.get(list.size()-1)-list.get(0);
            }
        }
        pw.println(max);
        pw.close();
    }
}