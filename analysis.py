import numpy as np
import os
import math
from sklearn.cluster import KMeans
from tqdm import tqdm
import queue

from shard import BFS_WALK, get_graph, find_unassigned_neighbors

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAND_SHARD_OPT = os.path.join(ROOT_DIR, "random_cluster", "optimal")
RAND_SHARD_4MIN = os.path.join(ROOT_DIR, "random_cluster", "4min")
KMEANS_CLUSTER = os.path.join(ROOT_DIR, "kmeans_cluster")
BFS_CLUSTER = os.path.join(ROOT_DIR, "bfs_walk")


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


def kmeans_method(g: int, A: np.ndarray) -> np.ndarray:
    kmeans = KMeans(n_clusters=g, random_state=0, n_init="auto")
    return kmeans.fit_predict(A)


def graph_traversal_method(
    n: int, c: int, A: np.ndarray, frontier: queue.Queue | queue.LifoQueue
) -> np.ndarray:
    assignment = np.zeros(n, dtype=int)
    unassigned = np.ones(n, dtype=int)
    g_idx = 0
    curr_cap = 0
    assigned = 0
    frontier.put(0)
    with tqdm(total=n) as pbar:
        while assigned < n:
            node = frontier.get()

            assignment[node] = g_idx
            assigned += 1
            pbar.update(1)
            unassigned[node] = 0
            curr_cap += 1

            if curr_cap == c:
                g_idx += 1
                curr_cap = 0

            for neighbor in find_unassigned_neighbors(node, unassigned, A):
                frontier.put(neighbor)

    return assignment


def bfs_method(n: int, c: int, A: np.ndarray) -> np.ndarray:
    print("Doing BFS...")
    frontier = queue.Queue()
    assignment = graph_traversal_method(n, c, A, frontier)
    return assignment


def dfs_method(n: int, c: int, A: np.ndarray) -> np.ndarray:
    print("Doing DFS...")
    frontier = queue.LifoQueue()
    assignment = graph_traversal_method(n, c, A, frontier)
    return assignment


def greedy_assign_node(
    node: int, assignments: np.ndarray, capacities: np.ndarray, A: np.ndarray
) -> int:
    neighbors = np.nonzero(A[node])[0]
    neighbors_ass = assignments[neighbors]
    neighbors_ass_real = neighbors_ass != -1
    neighbors_ass_clean = np.extract(neighbors_ass_real, neighbors_ass)

    mem_banks, counts = np.unique(neighbors_ass_clean, return_counts=True)
    choices_with_counts = sorted(
        zip(mem_banks, counts), key=lambda x: x[1], reverse=True
    )
    choices = map(lambda x: x[0], choices_with_counts)

    viable_choices = filter(lambda x: capacities[x] > 0, choices)
    res = next(viable_choices, None)
    if res is not None:
        return res
    return np.where(capacities > 0)[0][0]


def greedy_heuristic(n: int, g: int, c: int, A: np.ndarray) -> np.ndarray:
    print("Doing greedy...")
    assignments = -np.ones(n, dtype=int)
    capacities = np.full(g, c, dtype=int)

    assigned = 0
    with tqdm(total=n) as pbar:
        while assigned < n:
            node = np.where(assignments == -1)[0][0]
            ass = greedy_assign_node(node, assignments, capacities, A)
            assignments[node] = ass
            capacities[ass] -= 1
            assigned += 1
            pbar.update(1)

    return assignments


rand_cluster_opt = np.load(os.path.join(RAND_SHARD_OPT, "assignment.npy"))
rand_cluster_4min = np.load(os.path.join(RAND_SHARD_4MIN, "assignment.npy"))
bfs_cluster = np.load(os.path.join(BFS_WALK, "assignment.npy"))
# kmeans_cluster = np.load(os.path.join(KMEANS_CLUSTER, "assignment.npy"))
A, n = get_graph()
g = 128
c = math.ceil(n / g)
random_split = random_split_traffic(n, g, c)
bfs = bfs_method(n, c, A)
dfs = dfs_method(n, c, A)
kmeans = kmeans_method(g, A)
greedy = greedy_heuristic(n, g, c, A)

rand_cluster_opt_traffic = eval_traffic(rand_cluster_opt, A)
rand_cluster_4min_traffic = eval_traffic(rand_cluster_4min, A)
rand_split_traffic = eval_traffic(random_split, A)
kmeans_traffic = eval_traffic(kmeans, A)
# kmeans_cluster_traffic = eval_traffic(kmeans_cluster, A)
bfs_traffic = eval_traffic(bfs, A)
dfs_traffic = eval_traffic(dfs, A)
bfs_cluster_traffic = eval_traffic(bfs_cluster, A)
greedy_traffic = eval_traffic(greedy, A)

print("Network traffic:")
print(f"\t random split {rand_split_traffic}")
print(f"\t random cluster with solver {rand_cluster_opt_traffic}")
print(f"\t random cluster with solver (4 mins max) {rand_cluster_4min_traffic}")
print(f"\t kmeans split {kmeans_traffic}")
# print(f"\tkmeans cluster split {kmeans_cluster_traffic}")
print(f"\t BFS split {bfs_traffic}")
print(f"\t DFS split {dfs_traffic}")
print(f"\t BFS cluster {bfs_cluster_traffic}")
print(f"\t Greedy {greedy_traffic}")

random_cluster_improvement = rand_split_traffic / rand_cluster_opt_traffic
random_cluster_4min_improvement = rand_split_traffic / rand_cluster_4min_traffic
kmeans_improvement = rand_split_traffic / kmeans_traffic
# kmeans_cluster_improvement = rand_split_traffic / kmeans_cluster_traffic
bfs_improvement = rand_split_traffic / bfs_traffic
dfs_improvement = rand_split_traffic / dfs_traffic
bfs_cluster_improvement = rand_split_traffic / bfs_cluster_traffic
greedy_improvement = rand_split_traffic / greedy_traffic

print(f"random cluster improvement: {random_cluster_improvement}")
print(f"random cluster improvement (4 min max): {random_cluster_4min_improvement}")
print(f"kmeans improvement: {kmeans_improvement}")
# print(f"kmeans cluster improvement: {kmeans_cluster_improvement}")
print(f"BFS improvement: {bfs_improvement}")
print(f"DFS improvement: {dfs_improvement}")
print(f"BFS cluster improvement: {bfs_cluster_improvement}")
print(f"Greedy improvement: {greedy_improvement}")
