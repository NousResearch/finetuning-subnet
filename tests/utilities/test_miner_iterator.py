from utilities.miner_iterator import MinerIterator
import unittest


class TestMinerIterator(unittest.TestCase):
    def test_miner_uids_are_sorted(self):
        """Creates a MinerIterator with unsorted miner UIDs and verifies that the miner UIDs are sorted."""
        uids = [2, 5, 1, 0]
        iterator = MinerIterator(uids)

        # The iterator starts at a random position. Move it until we're pointing to 0.
        while iterator.peek() != 0:
            next(iterator)

        # Now verify the UIDs are iterated in sorted order.
        iterated_uids = [next(iterator) for _ in range(len(uids))]
        self.assertEqual(iterated_uids, sorted(uids))

    def test_iterator_is_infinite(self):
        """Creates a MinerIterator and verifies calling it more times than the number of miner UIDs cycles the UIDs."""
        uids = [3, 2, 1]
        expected = [1, 2, 3] * 10
        iterator = MinerIterator(uids)
        iterated_uids = [next(iterator) for _ in range(30)]
        self.assertEqual(sorted(iterated_uids), sorted(expected))

    def test_peek(self):
        """Creates a MinerIterator and verifies that peek returns the next UID without advancing the iterator."""
        uids = [1, 2, 3]
        iterator = MinerIterator(uids)

        peeked = iterator.peek()
        self.assertEqual(peeked, iterator.peek())
        self.assertEqual(peeked, next(iterator))
        self.assertNotEqual(peeked, iterator.peek())

    def test_set_miner_uids(self):
        """Verifies the iterator position is maintained when the miner UIDs are updated."""
        initial_miner_uids = [1, 2, 3, 4, 5]
        iterator = MinerIterator(initial_miner_uids)

        # Advance the iterator so it should now point to 3
        # The iterator starts at a random position. Advance it until it returns 2.
        while next(iterator) != 2:
            pass

        iterator.set_miner_uids([1, 4, 6])

        # Verify the iterator picks up from the next UID greater than or equal to 3.
        self.assertEqual(next(iterator), 4)
        self.assertEqual(next(iterator), 6)
        self.assertEqual(next(iterator), 1)

    def test_set_miner_uids_edge_case(self):
        """Verifies the iterator position is reset when the miner UIDs are updated and the current position is no longer valid."""
        # Create a MinerIterator with initial miner UIDs
        initial_miner_uids = [1, 2, 3, 4, 5]
        iterator = MinerIterator(initial_miner_uids)

        # Advance the iterator so it should now point to 5
        while iterator.peek() != 5:
            next(iterator)

        iterator.set_miner_uids([1, 2, 3, 4])

        self.assertEqual(next(iterator), 1)
        self.assertEqual(next(iterator), 2)
        self.assertEqual(next(iterator), 3)
        self.assertEqual(next(iterator), 4)


if __name__ == "__main__":
    unittest.main()
