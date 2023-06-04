import numpy as np
import random

# does not consider remaining capacity when assigning nodes
def greedy1(n: int, g: int, c: int, A: np.ndarray) -> np.ndarray:
    UNASSIGNED = -1
    assignment = np.full(n, UNASSIGNED)
    assign_order = np.random.permutation(n)
    remaining_cap = np.full(g, c)
    for i in range(n):
        # assign node assign_order[i]
        node_id = assign_order[i]
        neighbors = np.nonzero(A[node_id])[0]
        connectivity_count = np.zeros(g, dtype=np.int32)
        for neighbor_id in range(neighbors.size):
            if assignment[neighbor_id] != UNASSIGNED:
                connectivity_count[assignment[neighbor_id]] += 1
        max_connectivity = -1  # any negative number will be ok
        for unit_id in range(g):
            if remaining_cap[unit_id] > 0:
                if connectivity_count[unit_id] > max_connectivity:
                    max_connectivity = connectivity_count[unit_id]
        candidates = []
        for unit_id in range(g):
            if remaining_cap[unit_id] > 0:
                if connectivity_count[unit_id] == max_connectivity:
                    candidates.append(unit_id)
        assert len(candidates) > 0
        assignment[i] = random.choice(candidates)
        remaining_cap[assignment[i]] -= 1
    return assignment


# considers remaining capacity when assigning nodes
def greedy2(n: int, g: int, c: int, A: np.ndarray) -> np.ndarray:
    UNASSIGNED = -1
    assignment = np.full(n, UNASSIGNED)
    assign_order = np.random.permutation(n)
    remaining_cap = np.full(g, c)
    for i in range(n):
        # assign node assign_order[i]
        node_id = assign_order[i]
        neighbors = np.nonzero(A[node_id])[0]
        connectivity_count = np.zeros(g, dtype=np.int32)
        for neighbor_id in range(neighbors.size):
            if assignment[neighbor_id] != UNASSIGNED:
                connectivity_count[assignment[neighbor_id]] += 1
        max_connectivity = -1  # any negative number will be ok
        max_remaining_cap = -1  # any negative number will be ok
        for unit_id in range(g):
            if remaining_cap[unit_id] > 0:
                if connectivity_count[unit_id] > max_connectivity:
                    max_connectivity = connectivity_count[unit_id]
                    max_remaining_cap = remaining_cap[unit_id]
                elif connectivity_count[unit_id] == max_connectivity \
                        and remaining_cap[unit_id] > max_remaining_cap:
                    max_remaining_cap = remaining_cap[unit_id]
        candidates = []
        for unit_id in range(g):
            if remaining_cap[unit_id] > 0:
                if connectivity_count[unit_id] == max_connectivity \
                        and remaining_cap[unit_id] == max_remaining_cap:
                    candidates.append(unit_id)
        assert len(candidates) > 0
        assignment[i] = random.choice(candidates)
        remaining_cap[assignment[i]] -= 1
    return assignment
