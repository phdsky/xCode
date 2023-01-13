import edu.princeton.cs.algs4.WeightedQuickUnionUF;

public class Percolation {
    private final int size;
    private final int top;
    private final int bottom;

    private final WeightedQuickUnionUF uf;
    private final WeightedQuickUnionUF ufo;

    private int count;
    private boolean[] ufBlocked;

    // creates n-by-n grid, with all sites initially blocked
    public Percolation(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("Input error: n <= 0");
        }

        size = n;
        int sum = n * n;
        uf = new WeightedQuickUnionUF(sum + 1);
        ufo = new WeightedQuickUnionUF(sum + 2);

        top = sum;
        bottom = sum + 1;

        ufBlocked = new boolean[sum + 1];
        ufBlocked[top] = true;

        count = 0;
    }

    private int index(int row, int col) {
        checkRange(row, col);
        return (row - 1) * size + col - 1;
    }

    private void checkRange(int row, int col) {
        if (row < 1 || row > size || col < 1 || col > size) {
            throw new IllegalArgumentException("row or col out of range");
        }
    }

    // opens the site (row, col) if it is not open already
    public void open(int row, int col) {
        if (isOpen(row, col)) {
            return;
        }

        ufBlocked[index(row, col)] = true;
        count += 1;

        // up
        if (row == 1) {
            uf.union(index(row, col), top);
            ufo.union(index(row, col), top);
        }
        else if (row > 1 && isOpen(row - 1, col)) {
            uf.union(index(row - 1, col), index(row, col));
            ufo.union(index(row - 1, col), index(row, col));
        }

        // down
        if (row < size && isOpen(row + 1, col)) {
            uf.union(index(row + 1, col), index(row, col));
            ufo.union(index(row + 1, col), index(row, col));
        } else if (row == size) {
            ufo.union(index(row, col), bottom);
        }

        // left
        if (col - 1 >= 1 && isOpen(row, col - 1)) {
            uf.union(index(row, col - 1), index(row, col));
            ufo.union(index(row, col - 1), index(row, col));
        }

        // right
        if (col + 1 <= size && isOpen(row, col + 1)) {
            uf.union(index(row, col + 1), index(row, col));
            ufo.union(index(row, col + 1), index(row, col));
        }
    }

    // is the site (row, col) open?
    public boolean isOpen(int row, int col) {
        checkRange(row, col);

        return ufBlocked[index(row, col)];
    }

    // is the site (row, col) full?
    public boolean isFull(int row, int col) {
        if (!isOpen(row, col)) {
            return false;
        }

        return uf.find(index(row, col)) == uf.find(top);
    }

    // returns the number of open sites
    public int numberOfOpenSites() {
        return count;
    }

    // does the system percolate?
    public boolean percolates() {
        return ufo.find(bottom) == ufo.find(top);
    }

    // test client (optional)
    public static void main(String[] args) {
        System.out.println("Test client (optional)");
    }
}