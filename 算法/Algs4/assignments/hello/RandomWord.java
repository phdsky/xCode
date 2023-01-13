/* *****************************************************************************
 *  Name:              Ada Lovelace
 *  Coursera User ID:  123456
 *  Last modified:     October 16, 1842
 **************************************************************************** */

import edu.princeton.cs.algs4.StdIn;
import edu.princeton.cs.algs4.StdOut;
import edu.princeton.cs.algs4.StdRandom;

public class RandomWord {
    public static void main(String[] args) {
        String champion = null;
        double i = 0.0;

        while (!StdIn.isEmpty()) {
            String s = StdIn.readString();
            i += 1;
            double probability = 1 / i;
            if (StdRandom.bernoulli(probability)) {
                champion = s;
            }
        }

        StdOut.println(champion);
    }
}
