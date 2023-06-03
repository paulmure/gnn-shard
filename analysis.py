import numpy as np
import os
import math

from shard import get_graph

RAND_SHARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_cluster")


def eval_traffic(assignments: np.ndarray, A: np.ndarray) -> int:
    nzs = np.nonzero(A)
    left = assignments[nzs[0]]
    right = assignments[nzs[1]]
    return np.sum(left != right)


def random_split_traffic(n: int, g: int, c: int) -> np.ndarray:
    perm = np.random.permutation(range(n))
    assignments = np.zeros(n, dtype=int)
    g_idx = 0
    for start in range(0, n, c):
        end = min(start + c, n)
        assignments[perm[start:end]] = np.full(end - start, g_idx, dtype=int)
        g_idx += 1
    assert g_idx <= g
    return assignments


rand_cluster = np.load(os.path.join(RAND_SHARD, "assignment.npy"))
A, n = get_graph()
g = 128
c = math.ceil(n / g)
random_split = random_split_traffic(n, g, c)

rand_cluster_traffic = eval_traffic(rand_cluster, A)
rand_split_traffic = eval_traffic(random_split, A)

print(f"Network traffic:")
print(f"\trandom cluster with solver {rand_cluster_traffic}")
print(f"\trandom split {rand_split_traffic}")
print(f"improvement: {rand_split_traffic / rand_cluster_traffic}")
