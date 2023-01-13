import edu.princeton.cs.algs4.StdOut;
import edu.princeton.cs.algs4.StdRandom;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class RandomizedQueue<Item> implements Iterable<Item> {
    private int size;
    private Item[] items;

    // construct an empty randomized queue
    public RandomizedQueue() {
        size = 0;
        items = (Item[]) new Object[1];
    }

    // is the randomized queue empty?
    public boolean isEmpty() {
        return (size == 0);
    }

    // return the number of items on the randomized queue
    public int size() {
        return size;
    }

    private void resize(int capacity) {
        Item[] copy = (Item[]) new Object[capacity];
        for (int i = 0; i < size; i++) {
            copy[i] = items[i];
        }
        items = copy;
    }

    // add the item
    public void enqueue(Item item) {
        if (item == null) {
            throw new IllegalArgumentException();
        }

        if (size == items.length) {
            resize(2 * size);
        }
        items[size++] = item;
    }

    // remove and return a random item
    public Item dequeue() {
        if (size == 0) {
            throw new NoSuchElementException();
        }

        int id = StdRandom.uniform(size);
        Item item = items[id];

        items[id] = items[--size];
        items[size] = null;

        if (size > 0 && size == items.length / 4) {
            resize(items.length / 2);
        }

        return item;
    }

    // return a random item (but do not remove it)
    public Item sample() {
        if (size == 0) {
            throw new NoSuchElementException();
        }

        int id = StdRandom.uniform(size);
        Item item = items[id];

        return item;
    }

    // return an independent iterator over items in random order
    public Iterator<Item> iterator() {
        return new RandomizedQueueIterator();
    }

    private class RandomizedQueueIterator implements Iterator<Item> {
        private final Item[] iteratorItems;
        private int current = 0;

        private RandomizedQueueIterator() {
            iteratorItems = (Item[]) new Object[size];
            for (int i = 0; i < size; ++i) {
                iteratorItems[i] = items[i];
            }
            StdRandom.shuffle(iteratorItems);
        }

        public boolean hasNext() {
            return (current < size);
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public Item next() {
            if (current == size) {
                throw new NoSuchElementException();
            }

            return iteratorItems[current++];
        }
    }

    // unit testing (required)
    public static void main(String[] args) {
        RandomizedQueue<String> randomizedQueue = new RandomizedQueue<String>();
        StdOut.println(randomizedQueue.isEmpty());

        randomizedQueue.enqueue("Happy");
        randomizedQueue.enqueue("Chinese");
        randomizedQueue.enqueue("Lunar");
        randomizedQueue.enqueue("New");
        randomizedQueue.enqueue("Year");

        for (String s : randomizedQueue) {
            StdOut.println(s);
        }
        StdOut.println(randomizedQueue.size());

        StdOut.println(randomizedQueue.dequeue());
        StdOut.println(randomizedQueue.dequeue());
        StdOut.println(randomizedQueue.sample());
        StdOut.println(randomizedQueue.size());

        for (String s : randomizedQueue) {
            StdOut.println(s);
        }
    }

}