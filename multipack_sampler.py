from typing import Optional, List

import torch.distributed as dist
from torch.utils.data import Sampler

import numpy as np
import numba


@numba.njit
def lpt_check(heap: np.ndarray, A: np.ndarray, c: int, n: int):
    # LPT (Longest processing time first scheduling)
    # Time: O(|A| log |A| + |A| log n)
    """ Determine whether it's possible to assign A to n processors such that processor exceeds a maximum capacity c.
    heap size = n + 1, heap[0] is not used because the heap implementation is 1-indexed rather than 0-indexed. 
    This is a common approach in implementing heaps for easier calculation of parent and child nodes.
    **For single processor based bin packing. n=1, and this is just to find whether the sum of A can fit into the single bin.**
    Hence, this function works the best for larger n otherwise it is just too simple. 
    """

    A = np.sort(A)[::-1]
    heap.fill(0)
    for size in A:
        # Put into smallest element
        heap[1] += size 
        if heap[1] > c: 
            # heap[1] is always the smallest element in the heap. this ensure that we are using all bins maximally. 
            return False

        # Heapify (Sink). This part is just used to maintain the heap architecture (order). 
        # https://stackoverflow.com/questions/20397674/replacing-element-in-min-heap
        u = 1
        while (u << 1) <= n:
            v = u << 1  # lch
            rch = (u << 1) | 1
            if rch <= n and heap[rch] < heap[v]:
                v = rch
            
            if heap[u] <= heap[v]:
                break

            heap[u], heap[v] = heap[v], heap[u]
            u = v

    return True


@numba.njit
def lpt_with_result(heap: np.ndarray, A: np.ndarray, n: int, start_index: int, rank: int):
    # LPT (Longest processing time first scheduling)
    # Time: O(|A| log |A| + |A| log n)

    result = []

    indices = np.argsort(A)[::-1]
    A = A[indices]

    heap.fill(0)
    heap_id = np.arange(-1, n, dtype=A.dtype)
    for idx, size in enumerate(A):
        # Put into smallest element
        heap[1] += size
        if heap_id[1] == rank:
            result.append(start_index + indices[idx])

        # Heapify (Sink)
        # https://stackoverflow.com/questions/20397674/replacing-element-in-min-heap
        u = 1
        while (u << 1) <= n:
            v = u << 1  # lch
            rch = (u << 1) | 1
            if rch <= n and heap[rch] < heap[v]:
                v = rch
            
            if heap[u] <= heap[v]:
                break

            heap[u], heap[v] = heap[v], heap[u]
            heap_id[u], heap_id[v] = heap_id[v], heap_id[u]
            u = v

    return result


@numba.njit
def allocate(lengths: np.ndarray, lengths_cumsum: np.ndarray, rank: int, c: int, n: int):
    # Dynamic batch allocator, binary search + LPT
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    heap = np.zeros(n + 1, dtype=lengths.dtype)

    s = 0
    start_index = 0
    result = []

    while True:
        # binary search [l, r)
        l = 1
        r = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        # find the maximum m such that the consecutive tasks in lengths[start_index: start_index + m] can fit into n processors .
        # when n=1 for single process based case, each step is just to check whether sum of tasks less than c.  
        while r - l > 1:
            m = (l + r) // 2
            if lpt_check(heap, lengths[start_index: start_index + m], c, n):
                l = m
            else:
                r = m

        # use length l
        if l < n:
            break  # Can't allocate each sequence to a single machine
            
        # here is basically the bin pack part: pack all elements into a SINGLE bin, for each processor n. 
        # as we can see, this algorithm try to pack consecutive elements into a single bin, maybe not the best bin pack algorithm 
        # TODO: think about other algorithm for single gpu case, such that the bin pack can be more effective. 
        batch = lpt_with_result(heap, lengths[start_index: start_index + l], n, start_index, rank) 

        start_index += l
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch)

    return result, s, len(result) * c * n


class MultipackDistributedBatchSampler(Sampler):
    """Unpadded length sampling using Multipack V2, for models with quadratic attention complexity.
       It also tries to evenly distribute the sequences using LPT, so that quadratic load is more balanced.

       Approximate (at most 1.33x ?) the optimal solution of the identical-machines scheduling problem, which is NP-hard.

       Time Complexity: O(n log n log k)
       n = maximum number of sequences per batch, k = number of nodes
    """

    def __init__(
        self,
        batch_max_length: int,
        lengths: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        # Get rank
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.batch_max_length = batch_max_length
        self.lengths = lengths
        assert isinstance(self.lengths, np.ndarray)

        self.epoch = 0

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        indices = np.random.Generator(np.random.Philox(seed=self.seed + self.epoch)).permutation(len(self.lengths))

        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, total_used, total_slots = allocate(lengths=lengths,
                                                    lengths_cumsum=lengths_cumsum,
                                                    rank=self.rank,
                                                    c=self.batch_max_length,
                                                    n=self.num_replicas)
        
        batches = [indices[batch] for batch in batches]

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots
        
        return batches
    
    def __iter__(self):
        batches = self.generate_batches(set_stats=True)
        return iter(batches)

    def num_batches(self):
        batches = self.generate_batches()
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots
