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

TIMEOUT = 60 * 6


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
    # return A, num_nodes.item()
    n = 128
    return A[:n, :n], n


# n -> number of nodes in graph
# g -> number of gpus
# capVec -> number of node that can fit in each gpu
# A -> Adjacency matrix
def get_sharad_model(n: int, g: int, cap_remaining: np.ndarray, A: np.ndarray):
    m = gp.Model("shard")
    # m.Params.OutputFlag = 0
    # m.Params.MIPGap = 0.08
    m.Params.Threads = 16
    m.Params.TimeLimit = TIMEOUT

    edges = np.nonzero(A)

    # S -> GPU/node assignment matrix
    S = m.addMVar(shape=(n, g), vtype=GRB.BINARY, name="S")
    # S = m.addMVar(shape=(n,), vtype=GRB.INTEGER, lb=0, ub=n - 1, name="S")

    onesG = np.ones(g, dtype=int)
    onesN = np.ones(n, dtype=int)

    m.addConstr((S @ onesG) == onesN, name="one-to-one")
    m.addConstr((onesN @ S) <= cap_remaining, name="capacity")

    # Maximize locality
    # objective = ((S @ S.T) * A).sum()

    # Minimize traffic
    traffics = []
    for i in range(edges[0].shape[0]):
        a, b = edges[0][i], edges[1][i]

        aux = m.addVars(g, vtype=GRB.INTEGER)
        for j in range(g):
            aux2 = m.addVar(lb=-1, ub=1)
            m.addConstr(aux2 == (S[a, j] - S[b, j]))
            m.addConstr(aux[j] == gp.abs_(aux2))
        traffics.append(gp.quicksum(aux))

    objective = gp.quicksum(traffics)

    m.setObjective(objective, GRB.MINIMIZE)

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


def kmeans_perm(n: int, g: int, A: np.ndarray) -> np.ndarray:
    prediction = KMeans(n_clusters=g, n_init="auto").fit_predict(A)
    pred_idx = zip(prediction, range(n))

    sorted_pred_idx = sorted(pred_idx, key=lambda x: x[0])
    perm = np.array(list(map(lambda x: x[1], sorted_pred_idx)))
    return perm


def main():
    # batch_size = int(sys.argv[1])
    path = sys.argv[2]

    A, n = get_graph()
    num_edges = A.sum()
    density = (num_edges / (n * n)) * 100
    print(f"{n} nodes with {A.sum()} edges, density = {density:.2f}%")

    batch_size = n
    g = 4
    c = math.ceil(n / g)

    print(f"g = {g}, c = {c}, batch_size = {batch_size}")

    perm = kmeans_perm(n, g, A)
    rand_cluster(n, g, c, A, batch_size, perm, path)
    # kmeans_cluster_method(n, g, c, A, batch_size)
    # random_walk_bfs(n, g, c, A, batch_size)


if __name__ == "__main__":
    main()
