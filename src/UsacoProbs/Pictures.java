package UsacoProbs;

import java.io.IOException;
import java.util.Scanner;

public class Pictures {

    public static void main(String[] args) throws IOException {
        Scanner r = new Scanner(System.in);

        long left = r.nextInt();
        long right = r.nextInt();

        if (left == 0L){
            if (right >= 2) {
                System.out.println(right - 1);
            }
            else {
                System.out.println(0);
            }
        }
        else if (left == 1L) {
            if (right >= 1) {
                System.out.println(right);
            }
            else {
                System.out.println(0);
            }
        }
        else {
            int count = 0;
            for (int i = 0; i < left+1; i++) {
                if (i <= 2) {
                    count += right-1+i;
                }
                else {
                    count += right+1;
                }
            }
            System.out.println(count);
        }
    }
}
