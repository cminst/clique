package UsacoProbs.bronze.cowsay;

import com.github.ricksbrown.cowsay.Cowsay;

public class cowsay {
    public static void main(String[] args) {
        String[] args2 = new String[]{"-l"};
        String[] args3 = new String[]{"-f", "kitty", "USACO book problems"};
        String result = Cowsay.say(args2);
        String result2 = Cowsay.say(args3);
        System.out.println(result);
        System.out.println(result2);
    }
}
