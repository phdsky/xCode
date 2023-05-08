import java.util.Stack;

import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.MinPQ;
import edu.princeton.cs.algs4.StdOut;

public class Solver {
    private Node currNode;
    private final boolean solvable;

    private class Node implements Comparable<Node> {
        private final int currMove;
        private final int priority;
        private final Node preNode;
        private final Board board;

        public Node(Board initial) {
            this.currMove = 0;
            this.board = initial;
            this.preNode = null;
            this.priority = board.manhattan();
        }

        public Node(Board currBoard, Node preNode) {
            this.currMove = preNode.currMove + 1;
            this.board = currBoard;
            this.preNode = preNode;
            this.priority = this.currMove + this.board.manhattan();
        }

        @Override
        public int compareTo(Node that) {
            if (this.priority > that.priority) {
                return 1;
            } else if (this.priority == that.priority) {
                return 0;
            } else {
                return -1;
            }
        }
    }

    // find a solution to the initial board (using the A* algorithm)
    public Solver(Board initial) {
        if (null == initial) {
            throw new IllegalArgumentException("This is a empty initail tiles!");
        }

        MinPQ<Node> headPQ = new MinPQ<Solver.Node>();
        headPQ.insert(new Node(initial));
        MinPQ<Node> tailPQ = new MinPQ<Solver.Node>();
        tailPQ.insert(new Node(initial.twin()));

        while (true) {
            currNode = aStarSearch(headPQ);
            if (currNode.board.isGoal()) {
                solvable = true;
                break;
            }
            if (aStarSearch(tailPQ).board.isGoal()) {
                solvable = false;
                break;
            }
        }
    }

    // add priority-node's neighbors to MinPQ but not it's parents node;
    private Node aStarSearch(MinPQ<Node> pq) {
        Node priorityNode = pq.delMin();
        for (Board neighbor : priorityNode.board.neighbors()) {
            if (priorityNode.preNode == null || !neighbor.equals(priorityNode.preNode.board)) {
                pq.insert(new Node(neighbor, priorityNode));
            }
        }

        return priorityNode;
    }

    // is the initial board solvable? (see below)
    public boolean isSolvable() {
        return solvable;
    }

    // min number of moves to solve initial board; -1 if unsolvable
    public int moves() {
        if (!isSolvable()) {
            return -1;
        } else {
            return currNode.currMove;
        }
    }

    // sequence of boards in a shortest solution; null if unsolvable
    public Iterable<Board> solution() {
        if (isSolvable()) {
            Stack<Board> invSolution = new Stack<Board>();
            Node node = currNode;
            while (node != null) {
                invSolution.push(node.board);
                node = node.preNode;
            }
            Stack<Board> solution = new Stack<Board>();
            while (!invSolution.isEmpty()) {
                solution.push(invSolution.pop());
            }

            return solution;
        } else {
            return null;
        }
    }

    // test client (see below) 
    public static void main(String[] args) {

        // create initial board from file
        In in = new In(args[0]);
        int n = in.readInt();
        int[][] tiles = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                tiles[i][j] = in.readInt();
        Board initial = new Board(tiles);

        // solve the puzzle
        Solver solver = new Solver(initial);

        // print solution to standard output
        if (!solver.isSolvable())
            StdOut.println("No solution possible");
        else {
            StdOut.println("Minimum number of moves = " + solver.moves());
            for (Board board : solver.solution())
                StdOut.println(board);
        }
    }
}