import numpy as np
import gurobipy as gp
from gurobipy import GRB
from ogb.linkproppred import PygLinkPropPredDataset
import math
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os

CWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_cluster")


# Returns -> (Adjacency matrix, number of nodes)
def get_graph():
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root="dataset/")
    edge_list = dataset.get_edge_split()["train"]["edge"]
    num_nodes = edge_list.flatten().max() + 1
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    edge_list_T = np.transpose(edge_list)
    # make the graph undirected
    A[edge_list_T[0], edge_list_T[1]] = 1
    A[edge_list_T[1], edge_list_T[0]] = 1
    return A, num_nodes


# n -> number of nodes in graph
# g -> number of gpus
# capVec -> number of node that can fit in each gpu
# A -> Adjacency matrix
def get_sharad_model(n: int, g: int, capVec: np.ndarray, A: np.ndarray):
    m = gp.Model("shard")
    m.Params.Threads = 16

    # S -> GPU/node assignment matrix
    S = m.addMVar(shape=(n, g), vtype=GRB.BINARY, name="S")

    onesG = np.ones(g, dtype=int)
    onesN = np.ones(n, dtype=int)

    m.addConstr((S @ onesG) == onesN, name="one-to-one")
    m.addConstr((onesN @ S) <= capVec, name="capacity")

    # Maximize locality
    objective = sum(((S @ S[i]) @ A[:, i]) for i in range(n))
    m.setObjective(objective, GRB.MAXIMIZE)

    return m, S


def rand_cluster(n: int, g: int, c: int, A: np.ndarray, batch_size: int):
    perm = np.random.permutation(np.array(range(n)))
    capVec = np.full(g, c, dtype=int)
    assignments = np.zeros(n, dtype=int)

    solver_times = []

    for start in tqdm(range(0, batch_size, batch_size)):
        end = min(start + batch_size, n)
        batch = perm[start:end]
        subA = A[np.ix_(batch, batch)]

        model, S = get_sharad_model(batch_size, g, capVec, subA)
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        solver_times.append(end_time - start_time)
        nzs = np.nonzero(S.X)
        assignments[perm[nzs[0]]] = nzs[1]

    with open(os.path.join(CWD, "runtime.txt"), "w") as f:
        f.write("\n".join(list(map(lambda x: str(x), solver_times))))

    plt.xlabel("Batch Iteration")
    plt.ylabel("Solver Runtime")
    plt.title("Solver Runtime with Random Cluter Assignment")
    plt.plot(range(len(solver_times)), solver_times)
    plt.savefig(os.path.join(CWD, "runtime.png"))

    np.save(os.path.join(CWD, "assignment"), assignments)


A, n = get_graph()
num_edges = A.sum()
print(f"{n} nodes with {A.sum()} edges, density = {(num_edges/(n*n))*100:.2f}%")

g = 128
c = math.ceil(A.sum() / g)
batch_size = 128

rand_cluster(n, g, c, A, batch_size)
