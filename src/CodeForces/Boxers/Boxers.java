package CodeForces.Boxers;

import java.util.*;

public class Boxers {

    public static void main(String[] args) {
        Scanner r = new Scanner(System.in);

        int nBoxers = r.nextInt();

        var boxersWeights = new ArrayList<Integer>();
        for (int i = 0; i < nBoxers; i++) boxersWeights.add(r.nextInt());
        Collections.sort(boxersWeights);
        var boxers = new ArrayList<ArrayList<Integer>>();
        var weights = new HashSet<Integer>();
        for (int i = 0; i < nBoxers; i++) {
            boxers.add(new ArrayList<>());
            var boxer = boxersWeights.get(i);
            if (boxer != 1) {
                boxers.get(i).add(boxer - 1);
            }
            boxers.get(i).add(boxer);
            boxers.get(i).add(boxer + 1);
            weights.add(boxer);
        }
        if (weights.size() == nBoxers) System.out.println(nBoxers);
        else {
            var result = new HashMap<Integer, Integer>();
            for (ArrayList<Integer> boxer : boxers) {
                for (Integer integer : boxer) {
                    if (result.get(integer) == null) {
                        result.put(integer, 1);
                        break;
                    }
                }
            }
            System.out.println(result.size());
        }
    }
}

