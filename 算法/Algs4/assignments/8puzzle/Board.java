import java.util.ArrayList;

public class Board {
    private static final int BLANK = 0;
    private final int n;
    private final int[][] block;

    // create a board from an n-by-n array of tiles,
    // where tiles[row][col] = tile at (row, col)
    public Board(int[][] tiles) {
        if (null == tiles) {
            throw new IllegalArgumentException("The tiles is null");
        }

        n = tiles.length;
        this.block = new int[n][n];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                block[row][col] = tiles[row][col];
            }
        }
    }

    // string representation of this board
    public String toString() {
        StringBuilder strBuilder = new StringBuilder();
        strBuilder.append(n + "\n");
        
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                strBuilder.append(" " + block[row][col]);
            }
            strBuilder.append("\n");
        }

        String string = strBuilder.toString();
        return string;
    }

    // board dimension n
    public int dimension() {
        return n;
    }

    // number of tiles out of place
    public int hamming() {
        int distance = 0;
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                if (row*n + col + 1 != block[row][col] && 
                   (row*n + col + 1) < n*n) {
                    distance++;
                }
            }
        }

        return distance;
    }

    // sum of Manhattan distances between tiles and goal
    public int manhattan() {
        int distance = 0;
        int rightRow, rightCol, tileDistance;
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                if (BLANK == block[row][col]) {
                    continue;
                }
                rightRow = (block[row][col] - 1) / n;
                rightCol = (block[row][col] - 1) % n;
                tileDistance = Math.abs(rightRow - row) + Math.abs(rightCol - col);
                distance += tileDistance;
            }
        }

        return distance;
    }

    // is this board the goal board?
    public boolean isGoal() {
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                if (row*n + col + 1 != block[row][col] &&
                   (row*n + col + 1) < n*n) {
                    return false;
                }
            }
        }

        return true;
    }

    // does this board equal y?
    public boolean equals(Object y) {
        if (y == this) return true;
        if (y == null) return false;
        if (this.getClass() != y.getClass()) {
            return false;
        }

        Board that = (Board) y;
        if (this.dimension() != that.dimension()) {
            return false;
        }
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                if (this.block[row][col] != that.block[row][col]) {
                    return false;
                }
            }
        }

        return true;
    }

    // all neighboring boards
    public Iterable<Board> neighbors() {
        ArrayList<Board> boards = new ArrayList<>();
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                if (BLANK == block[row][col]) {
                    // left, right, up, down
                    if (row > 0) {
                        boards.add(exch(row, col, row - 1, col));
                    }
                    if (row < n - 1) {
                        boards.add(exch(row, col, row + 1, col));
                    }
                    if (col > 0) {
                        boards.add(exch(row, col, row, col - 1));
                    }
                    if (col < n - 1) {
                        boards.add(exch(row, col, row, col + 1));
                    }
                }
            }
        }

        return boards;
    }

    // a board that is obtained by exchanging any pair of tiles
    public Board twin() {
        int row = 0, col = 0, exRow = 1, exCol = 1;
        if (BLANK == block[row][col]) {
            return exch(row, col + 1, exRow, exCol);
        }
        if (BLANK == block[exRow][exCol]) {
            return exch(row, col, exRow - 1, exCol);
        }

        return exch(row, col, exRow, exCol);
    }

    private Board exch(int row, int col, int exRow, int exCol) {
        int[][] newTiles = copy();
        int swap = block[exRow][exCol];
        newTiles[exRow][exCol] = newTiles[row][col];
        newTiles[row][col] = swap;
        
        return new Board(newTiles);
    }

    private int[][] copy() {
        int[][] newTiles = new int[n][n];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                newTiles[row][col] = block[row][col];
            }
        }
        
        return newTiles;
    }

    // unit testing (not graded)
    public static void main(String[] args) {

    }

}