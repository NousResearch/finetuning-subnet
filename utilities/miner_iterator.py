import bisect
import copy
import threading
from typing import List

import random


class MinerIterator:
    """A thread safe infinite iterator to cyclically enumerate the current set of miner UIDs.

    Why? To perform miner evaluations, the validator will enumerate through the miners in order to help ensure
    each miner is evaluated at least once per epoch.
    """

    def __init__(self, miner_uids: List[int]):
        self.miner_uids = sorted(copy.deepcopy(miner_uids))
        # Start the index at a random position. This helps ensure that miners with high UIDs aren't penalized if
        # the validator restarts frequently.
        self.index = random.randint(0, len(self.miner_uids) - 1)
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self) -> int:
        with self.lock:
            if len(self.miner_uids) == 0:
                # This iterator should be infinite. If there are no miner UIDs, raise an error.
                raise IndexError("No miner UIDs.")

            uid = self.miner_uids[self.index]
            self.index += 1
            if self.index >= len(self.miner_uids):
                self.index = 0
            return uid

    def peek(self) -> int:
        """Returns the next miner UID without advancing the iterator."""
        with self.lock:
            if len(self.miner_uids) == 0:
                # This iterator should be infinite. If there are no miner UIDs, raise an error.
                raise IndexError("No miner UIDs.")

            return self.miner_uids[self.index]

    def set_miner_uids(self, miner_uids: List[int]):
        """Updates the miner UIDs to iterate.

        The iterator will be updated to the first miner uid that is greater than or equal to UID that would be next
        returned by the iterator. This helps ensure that frequent updates to the miner_uids does not cause too much
        churn in the sequence of UIDs returned by the iterator.
        """
        sorted_uids = sorted(copy.deepcopy(miner_uids))
        with self.lock:
            next_uid = self.miner_uids[self.index]
            new_index = bisect.bisect_left(sorted_uids, next_uid)
            if new_index >= len(sorted_uids):
                new_index = 0
            self.index = new_index
            self.miner_uids = sorted_uids
