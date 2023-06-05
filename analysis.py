import numpy as np
import os
import math
from tqdm import tqdm
import queue

from shard import get_graph, find_unassigned_neighbors
from greedy import greedy1, greedy2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAND_SHARD = os.path.join(ROOT_DIR, "random_cluster")
BFS_CLUSTER = os.path.join(ROOT_DIR, "bfs_walk")


def calculate_traffic(assignments: np.ndarray, A: np.ndarray) -> int:
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
            if not unassigned[node]:
                continue

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


def validate_assignment(n: int, g: int, c: int, assignment: np.ndarray, name: str):
    if assignment.shape[0] != n:
        raise Exception(f"{name} did not have the right number of node")

    mem_banks, counts = np.unique(assignment, return_counts=True)
    if np.max(mem_banks) >= g or np.min(mem_banks) < 0:
        raise Exception(f"{name} used a memory bank index that is not allowed")
    if np.max(counts) > c:
        raise Exception(f"{name} put too many nodes in a single bank")


def eval_assignment(
    n: int,
    g: int,
    c: int,
    baseline: int,
    A: np.ndarray,
    assignment: np.ndarray,
    name: str,
):
    validate_assignment(n, g, c, assignment, name)
    traffic = calculate_traffic(assignment, A)
    improvement = baseline / traffic
    print(f"{name}: traffic = {traffic}, {improvement:.2f} times over baseline")


def eval_all(
    n: int,
    g: int,
    c: int,
    random: np.ndarray,
    assignments: list[tuple[np.ndarray, str]],
    A: np.ndarray,
):
    baseline = calculate_traffic(random, A)
    list(
        map(
            lambda ass: eval_assignment(n, g, c, baseline, A, ass[0], ass[1]),
            assignments,
        )
    )


def main():
    A, n = get_graph()
    g = 128
    c = math.ceil(n / g)

    random_split = random_split_traffic(n, g, c)

    assignments = []

    for dir in os.listdir(RAND_SHARD):
        path = os.path.join(RAND_SHARD, dir, "assignment.npy")
        assignment = np.load(path)
        assignments.append((assignment, dir))

    bfs = bfs_method(n, c, A)
    dfs = dfs_method(n, c, A)

    greedy = greedy_heuristic(n, g, c, A)
    greedy1_ass = greedy1(n, g, c, A)
    greedy2_ass = greedy2(n, g, c, A)

    assignments = assignments + [
        (bfs, "BFS"),
        (dfs, "DFS"),
        (greedy, "Greedy"),
        (greedy1_ass, "Greedy 1"),
        (greedy2_ass, "Greedy 2"),
    ]

    eval_all(n, g, c, random_split, assignments, A)


if __name__ == "__main__":
    main()
