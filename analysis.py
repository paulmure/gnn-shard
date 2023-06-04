import numpy as np
import os
import math
from sklearn.cluster import KMeans

from shard import get_graph

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAND_SHARD = os.path.join(ROOT_DIR, "random_cluster")
KMEANS_CLUSTER = os.path.join(ROOT_DIR, "kmeans_cluster")


def eval_traffic(assignments: np.ndarray, A: np.ndarray) -> int:
    nzs = np.nonzero(A)
    left = assignments[nzs[0]]
    right = assignments[nzs[1]]
    return np.sum(left != right)


def random_split_traffic(n, g: int, c: int) -> np.ndarray:
    perm = np.random.permutation(range(n))
    assignments = np.zeros(n, dtype=int)
    g_idx = 0
    for start in range(0, n, c):
        end = min(start + c, n)
        assignments[perm[start:end]] = np.full(end - start, g_idx, dtype=int)
        g_idx += 1
    assert g_idx <= g
    return assignments


def kmeans_method(g: int, A: np.ndarray) -> np.ndarray:
    kmeans = KMeans(n_clusters=g, random_state=0, n_init="auto")
    return kmeans.fit_predict(A)


rand_cluster = np.load(os.path.join(RAND_SHARD, "assignment.npy"))
kmeans_cluster = np.load(os.path.join(KMEANS_CLUSTER, "assignment.npy"))
A, n = get_graph()
g = 128
c = math.ceil(n / g)
random_split = random_split_traffic(n, g, c)
kmeans = kmeans_method(g, A)

rand_cluster_traffic = eval_traffic(rand_cluster, A)
rand_split_traffic = eval_traffic(random_split, A)
kmeans_traffic = eval_traffic(kmeans, A)
kmeans_cluster_traffic = eval_traffic(kmeans_cluster, A)

print("Network traffic:")
print(f"\trandom cluster with solver {rand_cluster_traffic}")
print(f"\trandom split {rand_split_traffic}")
print(f"\tkmeans split {kmeans_traffic}")
print(f"\tkmeans cluster split {kmeans_cluster_traffic}")

random_cluster_improvement = rand_split_traffic / rand_cluster_traffic
kmeans_improvement = rand_split_traffic / kmeans_traffic
kmeans_cluster_improvement = rand_split_traffic / kmeans_cluster_traffic

print(f"random cluster improvement: {random_cluster_improvement}")
print(f"kmeans improvement: {kmeans_improvement}")
print(f"kmeans cluster improvement: {kmeans_cluster_improvement}")
