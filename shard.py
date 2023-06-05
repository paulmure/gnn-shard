import numpy as np
import queue
import sys
import gurobipy as gp
from gurobipy import GRB
from ogb.linkproppred import PygLinkPropPredDataset
import math
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
from sklearn.cluster import KMeans

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TIMEOUT = 60 * 4


# Returns -> (Adjacency matrix, number of nodes)
def get_graph() -> tuple[np.ndarray, int]:
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root="dataset/")
    edge_list = dataset.get_edge_split()["train"]["edge"]
    num_nodes = edge_list.flatten().max() + 1
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    edge_list_T = np.transpose(edge_list)
    # make the graph undirected
    A[edge_list_T[0], edge_list_T[1]] = 1
    A[edge_list_T[1], edge_list_T[0]] = 1
    return A, num_nodes.item()


# n -> number of nodes in graph
# g -> number of gpus
# capVec -> number of node that can fit in each gpu
# A -> Adjacency matrix
def get_sharad_model(n: int, g: int, cap_remaining: np.ndarray, A: np.ndarray):
    m = gp.Model("shard")
    m.Params.OutputFlag = 0
    m.Params.Threads = 16
    m.Params.TimeLimit = TIMEOUT

    # S -> GPU/node assignment matrix
    S = m.addMVar(shape=(n, g), vtype=GRB.BINARY, name="S")

    onesG = np.ones(g, dtype=int)
    onesN = np.ones(n, dtype=int)

    m.addConstr((S @ onesG) == onesN, name="one-to-one")
    m.addConstr((onesN @ S) <= cap_remaining, name="capacity")

    # Maximize locality
    objective = sum(((S @ S[i]) @ A[:, i]) for i in range(n))
    m.setObjective(objective, GRB.MAXIMIZE)

    return m, S


def rand_cluster(
    n: int, g: int, c: int, A: np.ndarray, batch_size: int, perm: np.ndarray, path
):
    if not os.path.exists(path):
        raise Exception("given path does not exists")

    cap_remaining = np.full(g, c, dtype=int)
    assignments = -np.ones(n, dtype=int)

    solver_times = []

    for start in tqdm(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch = perm[start:end]
        subA = A[np.ix_(batch, batch)]

        model, S = get_sharad_model(end - start, g, cap_remaining, subA)
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        solver_times.append(end_time - start_time)

        (subA_node_ids, mem_asses) = np.nonzero(S.X)
        assignments[perm[batch[subA_node_ids]]] = mem_asses

        mem_ids, counts = np.unique(mem_asses, return_counts=True)
        cap_remaining[mem_ids] -= counts

    with open(os.path.join(path, "runtime.txt"), "w") as f:
        f.write("\n".join(list(map(lambda x: str(x), solver_times))))

    plt.xlabel("Batch Iteration")
    plt.ylabel("Solver Runtime")
    plt.title("Solver Runtime")
    plt.plot(range(len(solver_times)), solver_times)
    plt.savefig(os.path.join(path, "runtime.png"))

    np.save(os.path.join(path, "assignment"), assignments)


# def kmeans_cluster_method(n: int, g: int, c: int, A: np.ndarray, batch_size: int):
#     n_clusters = math.ceil(n / batch_size)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
#     idxs = kmeans.fit_predict(A)
#     sorted_tups = sorted(zip(range(n), idxs), key=lambda x: x[1])
#     perm = np.array(list(map(lambda x: x[0], sorted_tups)))
#     rand_cluster(n, g, c, A, batch_size, perm, KMEANS_CLUSTER)


def find_unassigned_neighbors(
    node: int, unassigned: np.ndarray, A: np.ndarray
) -> np.ndarray:
    neighbors = np.nonzero(A[node])[0]
    available = np.nonzero(unassigned[neighbors])[0]
    res = neighbors[available]
    return res


def graph_traversal_permutation(
    n: int, A: np.ndarray, frontier: queue.Queue | queue.LifoQueue
) -> np.ndarray:
    perm = np.zeros(n, dtype=int)
    unassigned = np.ones(n, dtype=int)
    assigned = 0
    frontier.put(0)
    with tqdm(total=n) as pbar:
        while assigned < n:
            node = frontier.get()
            perm[assigned] = node
            assigned += 1
            pbar.update(1)
            unassigned[node] = 0

            for neighbor in find_unassigned_neighbors(node, unassigned, A):
                frontier.put(neighbor)

    return perm


# def random_walk_bfs(n: int, g: int, c: int, A: np.ndarray, batch_size: int):
#     frontier = queue.Queue()
#     perm = graph_traversal_permutation(n, A, frontier)
#     rand_cluster(n, g, c, A, batch_size, perm, BFS_WALK)


def main():
    batch_size = int(sys.argv[1])
    path = sys.argv[2]

    A, n = get_graph()
    num_edges = A.sum()
    density = (num_edges / (n * n)) * 100
    print(f"{n} nodes with {A.sum()} edges, density = {density:.2f}%")

    g = 128
    c = math.ceil(n / g)

    print(f"g = {g}, c = {c}, batch_size = {batch_size}")

    perm = np.random.permutation(np.array(range(n)))
    rand_cluster(n, g, c, A, batch_size, perm, path)
    # kmeans_cluster_method(n, g, c, A, batch_size)
    # random_walk_bfs(n, g, c, A, batch_size)


if __name__ == "__main__":
    main()
