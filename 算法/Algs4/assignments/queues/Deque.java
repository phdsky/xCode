import edu.princeton.cs.algs4.StdOut;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class Deque<Item> implements Iterable<Item> {
    private int size;
    private Node first;
    private Node last;

    private class Node {
        Item item;
        Node next;
        Node prev;
    }

    // construct an empty deque
    public Deque() {
        first = null;
        last = null;
        size = 0;
    }

    // is the deque empty?
    public boolean isEmpty() {
        return (size == 0);
    }

    // return the number of items on the deque
    public int size() {
        return size;
    }

    // add the item to the front
    public void addFirst(Item item) {
        if (item == null) {
            throw new IllegalArgumentException();
        }

        Node oldFirst = first;
        first = new Node();
        if (oldFirst == null) {
            first.next = null;
            last = first;
        } else {
            first.next = oldFirst;
            oldFirst.prev = first;
        }
        first.prev = null;
        first.item = item;
        size += 1;
    }

    // add the item to the back
    public void addLast(Item item) {
        if (item == null) {
            throw new IllegalArgumentException();
        }

        Node oldLast = last;
        last = new Node();
        if (oldLast == null) {
            last.prev = null;
            first = last;
        } else {
            last.prev = oldLast;
            oldLast.next = last;
        }
        last.next = null;
        last.item = item;
        size += 1;
    }

    // remove and return the item from the front
    public Item removeFirst() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }

        Node oldFirst = first;
        Item item = oldFirst.item;

        if (oldFirst.next == null) {
            first = null;
            last = first;
        } else {
            first = first.next;
            first.prev = null;
        }

        size -= 1;

        return item;
    }

    // remove and return the item from the back
    public Item removeLast() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }

        Node oldLast = last;
        Item item = oldLast.item;

        if (oldLast.prev == null) {
            last = null;
            first = last;
        } else {
            last = last.prev;
            last.next = null;
        }

        size -= 1;

        return item;
    }

    // return an iterator over items in order from front to back
    public Iterator<Item> iterator() {
        return new DequeIterator();
    }

    private class DequeIterator implements Iterator<Item> {
        private Node current = first;

        public boolean hasNext() {
            return (current != null);
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public Item next() {
            if (current == null) {
                throw new NoSuchElementException();
            }

            Item item = current.item;
            current = current.next;

            return item;
        }
    }

    // unit testing (required)
    public static void main(String[] args) {
        Deque<String> deque = new Deque<String>();
        StdOut.println(deque.isEmpty());

        deque.addFirst("Happy");
        deque.addLast("Chinese");
        deque.addLast("Lunar");
        deque.addLast("New");
        deque.addLast("Year");

        for (String s : deque) {
            StdOut.println(s);
        }
        StdOut.println(deque.size());

        deque.removeFirst();
        deque.removeLast();
        deque.removeLast();
        deque.removeLast();
        StdOut.println(deque.size());

        for (String s : deque) {
            StdOut.println(s);
        }
    }

}