import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FastCollinearPoints {
    private final List<LineSegment> lineSegments;

    public FastCollinearPoints(Point[] points) {
        checkNull(points);
        Point[] pointsCopy = points.clone();
        Arrays.sort(pointsCopy);
        checkDuplicate(pointsCopy);

        lineSegments = new ArrayList<>();

        int num = pointsCopy.length;
        for (int i = 0; i < num - 3; i++) {
            Arrays.sort(pointsCopy);
            Point p = pointsCopy[i];

            Arrays.sort(pointsCopy, p.slopeOrder());

            for (int first = 1, last = 2; last < num; last++) {
                while (last < num &&
                        Double.compare(p.slopeTo(pointsCopy[first]),
                                       p.slopeTo(pointsCopy[last])) == 0) {
                    last++;
                }

                if (last - first >= 3 && p.compareTo(pointsCopy[first]) < 0) {
                    lineSegments.add(new LineSegment(p, pointsCopy[last - 1]));
                }

                first = last;
            }
        }

    }    // finds all line segments containing 4 or more points

    public int numberOfSegments() {
        return lineSegments.size();
    }       // the number of line segments

    public LineSegment[] segments() {
        return lineSegments.toArray(new LineSegment[numberOfSegments()]);
    }               // the line segments

    private void checkNull(Point[] points) {
        if (points == null) {
            throw new IllegalArgumentException();
        }

        for (Point p : points) {
            if (p == null) {
                throw new IllegalArgumentException();
            }
        }
    }

    private void checkDuplicate(Point[] points) {
        for (int i = 0; i < points.length - 1; i++) {
            if (points[i].compareTo(points[i + 1]) == 0) {
                throw new IllegalArgumentException();
            }
        }
    }

    public static void main(String[] args) {
        // read the n points from a file
        In in = new In(args[0]);
        int n = in.readInt();
        Point[] points = new Point[n];
        for (int i = 0; i < n; i++) {
            int x = in.readInt();
            int y = in.readInt();
            points[i] = new Point(x, y);
        }

        // draw the points
        StdDraw.enableDoubleBuffering();
        StdDraw.setXscale(0, 32768);
        StdDraw.setYscale(0, 32768);
        for (Point p : points) {
            p.draw();
        }
        StdDraw.show();

        // print and draw the line segments
        FastCollinearPoints collinear = new FastCollinearPoints(points);
        for (LineSegment segment : collinear.segments()) {
            StdOut.println(segment);
            segment.draw();
        }
        StdDraw.show();
        StdOut.println(collinear.numberOfSegments());
    }
}
