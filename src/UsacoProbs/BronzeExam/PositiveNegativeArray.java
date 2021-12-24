package UsacoProbs.BronzeExam;

import java.io.IOException;
import java.util.Scanner;
import java.util.Stack;

public class PositiveNegativeArray {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(System.in);

        int size = r.nextInt();
        int[] array = new int[size];
        for (int i = 0; i < size; i++) {
            array[i] = r.nextInt();
        }

        Stack<Stack<Integer>> stack = new Stack<>();
        Stack<Integer> group = new Stack<>();
        for (int i = 0; i < array.length; i++) {
            if (array[i] != 0) {
                if (group.isEmpty() || (group.peek() < 0 && array[i] < 0) || (group.peek() > 0 && array[i] > 0)) {
                    group.push(array[i]);
                }
                else {
                    Stack<Integer> g = (Stack<Integer>) group.clone();
                    stack.push(g);
                    group.clear();
                    group.push(array[i]);
                }
            }
            else {
                Stack<Integer> g = (Stack<Integer>) group.clone();
                if (!g.isEmpty()) {
                    stack.push(g);
                }
                group.clear();
            }
        }
        if (!group.isEmpty()) {
            stack.push(group);
        }
        System.out.println(stack);
    }
}