import numpy as np
import gurobipy as gp
from gurobipy import GRB
from ogb.linkproppred import PygLinkPropPredDataset


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
# g -> number of GPUs/PMUs
# c -> number nodes that can fit in a single GPU/PMU
def get_sharad_model(n: int, g: int, c: int, A: np.ndarray):
    m = gp.Model("shard")
    m.Params.Threads = 16

    # S -> GPU/node assignment matrix
    S = m.addMVar(shape=(n, g), vtype=GRB.BINARY, name="S")

    onesG = np.ones(g, dtype=int)
    onesN = np.ones(n, dtype=int)
    capVec = np.full(g, c, dtype=int)

    m.addConstr((S @ onesG) == onesN, name="one-to-one")
    m.addConstr((onesN @ S) <= capVec, name="capacity")

    # Maximize locality
    objective = sum(((S @ S[i]) @ A[:, i]) for i in range(n))
    m.setObjective(objective, GRB.MAXIMIZE)

    return m, S


A, n = get_graph()

g = 2
c = 1

model, S = get_sharad_model(n, g, c, A)
model.optimize()
print(S.X)
